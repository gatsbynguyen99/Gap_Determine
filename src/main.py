import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import cv2

from pipeline.config import load_config
from pipeline.io_sources import iter_frames_from_folder, iter_frames_from_camera, iter_captures_from_dir
from pipeline.roi import apply_roi
from pipeline.preprocess import preprocess_to_gray
from pipeline.segment import segmentation_branch
from pipeline.quality import compute_quality, confidence_gate
from pipeline.scale import build_scale_estimator
from pipeline.report import write_results_csv, write_summary_json
from pipeline.viz import save_overlay

from pipeline.depth_measure import find_two_pipes_and_gap_from_depth, build_gap_mask_from_edges
from pipeline.measure import measure_gap_edges_from_mask

# ML helper imports (model + mask cleanup + component filter)
from pipeline.segment import OnnxSegModel, select_gap_component, _morph_cleanup


def make_run_dir(base_dir: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, ts)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ---------------------------------------------------------
    # Load ML model ONCE (secondary channel; depth is primary)
    # ---------------------------------------------------------
    ml_model = None

    # If you want ML always-on, load when model_path exists.
    # If you want ML only when segmentation.method == "ml_onnx", keep that condition.
    if cfg.segmentation.enabled and getattr(cfg.segmentation, "model_path", ""):
        if cfg.segmentation.model_path.strip():
            ml_model = OnnxSegModel(
                cfg.segmentation.model_path,
                input_size=cfg.segmentation.onnx_input_size,
                thresh=cfg.segmentation.onnx_thresh
            )

    run_dir = make_run_dir(cfg.output.dir)
    os.makedirs(run_dir, exist_ok=True)

    scale_est = build_scale_estimator(cfg.scale)

    results: List[Dict] = []
    overlay_count = 0

    # Temporal state for Layer D (camera stability)
    temporal_state: Dict = {}

    # Choose frame iterator
    if cfg.source.type == "folder":
        frame_iter = iter_frames_from_folder(cfg.source.path, cfg.source.glob, cfg.source.max_frames)
        mode = "rgb_only"
    elif cfg.source.type == "camera":
        frame_iter = iter_frames_from_camera(cfg.source.camera_index, cfg.source.max_frames)
        mode = "rgb_only"
    elif cfg.source.type == "captures":
        frame_iter = iter_captures_from_dir(cfg.source.path, cfg.source.max_frames)
        mode = "rgb_depth"
    else:
        raise ValueError(f"Unknown source.type: {cfg.source.type}")

    for item in frame_iter:
        if mode == "rgb_only":
            frame_id, frame = item
            depth = None
            meta = None
        else:
            frame_id, frame, depth, meta = item

        # ---------------------------------------------------------
        # 1) ROI
        # ---------------------------------------------------------
        frame_roi, (ox, oy) = apply_roi(frame, cfg.roi)

        roi_size = None
        if cfg.roi.enabled and cfg.roi.w > 0 and cfg.roi.h > 0:
            roi_size = (frame_roi.shape[1], frame_roi.shape[0])  # (w, h)

        # Raw grayscale for quality (blur metric)
        gray_raw = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)

        # Preprocessed grayscale for classical segmentation (CLAHE/blur)
        gray = preprocess_to_gray(frame_roi, cfg.preprocess)

        # Compute quality from RAW grayscale (do not overwrite later)
        quality = compute_quality(gray_raw)

        # ---------------------------------------------------------
        # 2) Depth ROI (offline captures)
        # ---------------------------------------------------------
        depth_roi = None
        if depth is not None:
            depth_roi, _ = apply_roi(depth, cfg.roi)

        # ---------------------------------------------------------
        # A) ML secondary output (compare ML vs depth)
        # ---------------------------------------------------------
        ml_gap_px = None
        ml_gap_x_left_roi = None
        ml_gap_x_right_roi = None
        ml_gap_x_left_full = None
        ml_gap_x_right_full = None
        ml_valid_ratio = 0.0
        ml_mask_coverage = None
        ml_mask01 = None

        if ml_model is not None:
            # 1) predict mask from ROI BGR (model expects 3-ch)
            ml_mask01 = ml_model.predict_mask01(frame_roi)  # should return 0/1 at ROI size

            # 2) cleanup
            ml_mask01 = _morph_cleanup((ml_mask01 * 255).astype("uint8"))
            ml_mask01 = (ml_mask01 == 255).astype("uint8")

            # 3) coverage check BEFORE measuring endpoints
            ml_mask_coverage = float(np.mean(ml_mask01))

            # If ML predicts almost everything or almost nothing -> invalid
            if 0.001 <= ml_mask_coverage <= 0.40:
                # 4) component filtering to keep only plausible gap component
                ml_mask01 = select_gap_component(
                    ml_mask01,
                    expected_min=cfg.segmentation.expected_gap_px_min,
                    expected_max=cfg.segmentation.expected_gap_px_max,
                    center_prior=None,
                    depth_map=None
                )

                # 5) measure endpoints from ML mask
                ml_gap_px, ml_info, ml_edges = measure_gap_edges_from_mask(ml_mask01, cfg.measurement)

                if ml_gap_px is not None:
                    ml_valid_ratio = float(ml_info.get("valid_ratio", 0.0))
                    ml_gap_x_left_roi = ml_edges.get("x_left_med", None)
                    ml_gap_x_right_roi = ml_edges.get("x_right_med", None)

                    if ml_gap_x_left_roi is not None:
                        ml_gap_x_left_full = int(ml_gap_x_left_roi + ox)
                    if ml_gap_x_right_roi is not None:
                        ml_gap_x_right_full = int(ml_gap_x_right_roi + ox)
            else:
                # invalid mask coverage
                ml_gap_px = None
                ml_valid_ratio = 0.0

            # Save ML mask (debug)
            if cfg.output.save_overlays and ml_mask01 is not None:
                ml_dir = os.path.join(run_dir, "ml_masks")
                os.makedirs(ml_dir, exist_ok=True)
                safe_id = os.path.basename(str(frame_id)).replace(":", "_").replace("/", "_").replace("\\", "_")
                cv2.imwrite(os.path.join(ml_dir, f"ml_mask_{safe_id}.png"), (ml_mask01 * 255).astype("uint8"))

        # ---------------------------------------------------------
        # B) Depth primary: pipes -> gap edges -> stable depth mask
        # ---------------------------------------------------------
        depth_gap_x_left_full = depth_gap_x_right_full = None
        pipe1_x0_full = pipe1_x1_full = None
        pipe2_x0_full = pipe2_x1_full = None
        near_depth_mm = None
        gap_px_depth = None

        mask01_depth = None
        depth_meas_info = {}

        if depth_roi is not None:
            depth_info = find_two_pipes_and_gap_from_depth(depth_roi, near_percentile=20.0, band_mm=60.0, min_area=500)

            if depth_info is not None:
                gxL = depth_info["gap_x_left"]      # ROI coords
                gxR = depth_info["gap_x_right"]     # ROI coords
                gap_px_depth = depth_info["gap_px_depth"]
                near_depth_mm = depth_info["near_depth_mm"]

                depth_gap_x_left_full = gxL + ox
                depth_gap_x_right_full = gxR + ox

                p1 = depth_info["pipe1"]  # (x0,x1,y0,y1) ROI coords
                p2 = depth_info["pipe2"]
                pipe1_x0_full, pipe1_x1_full = p1[0] + ox, p1[1] + ox
                pipe2_x0_full, pipe2_x1_full = p2[0] + ox, p2[1] + ox

                # Build stable depth-based gap mask (ROI coords)
                mask01_depth = build_gap_mask_from_edges(
                    H=depth_roi.shape[0],
                    W=depth_roi.shape[1],
                    x_left=gxL,
                    x_right=gxR,
                    y_min_frac=cfg.measurement.scanline_y_min,
                    y_max_frac=cfg.measurement.scanline_y_max
                )

                # Measure endpoints from depth mask (ROI coords)
                gap_px_mask, info_mask, edges_mask = measure_gap_edges_from_mask(mask01_depth, cfg.measurement)

                if gap_px_mask is not None:
                    depth_meas_info.update(info_mask)
                    depth_meas_info["gap_x_left_roi"] = edges_mask.get("x_left_med", None)
                    depth_meas_info["gap_x_right_roi"] = edges_mask.get("x_right_med", None)

        # ---------------------------------------------------------
        # C) Fallback classical segmentation (optional)
        # NOTE: only use this if depth fails and you want something.
        # ---------------------------------------------------------
        seg_gap_px = None
        seg_meas_info = {}
        seg_mask01 = None

        # Avoid running segmentation_branch in ML mode to prevent double-running ML ONNX.
        run_classical_seg = cfg.segmentation.enabled and (cfg.segmentation.method != "ml_onnx")

        if run_classical_seg:
            seg_gap_px, seg_meas_info, seg_mask01, temporal_state = segmentation_branch(
                gray,
                cfg.segmentation,
                cfg.measurement,
                temporal_state=temporal_state,
                depth_map=depth_roi,
                img_bgr=frame_roi
            )

        # ---------------------------------------------------------
        # D) Choose PRIMARY output: Depth if available, else segmentation fallback
        # ---------------------------------------------------------
        mask01_used = None
        gap_px = None
        meas_info = {}

        if mask01_depth is not None and depth_meas_info.get("gap_x_left_roi") is not None:
            mask01_used = mask01_depth
            gap_px = depth_meas_info.get("gap_px_median", None)
            meas_info = depth_meas_info
        else:
            mask01_used = seg_mask01
            gap_px = seg_gap_px
            meas_info = seg_meas_info

        # Save primary mask used
        if cfg.output.save_overlays and mask01_used is not None:
            mask_dir = os.path.join(run_dir, "masks")
            os.makedirs(mask_dir, exist_ok=True)
            safe_id = os.path.basename(str(frame_id)).replace(":", "_").replace("/", "_").replace("\\", "_")
            cv2.imwrite(os.path.join(mask_dir, f"mask_{safe_id}.png"), (mask01_used * 255).astype("uint8"))

        # Quality gate + scale
        if gap_px is None:
            ok = False
            conf = {"passes": False, "reasons": ["no_gap_found"]}
            gap_mm = None
        else:
            ok, conf = confidence_gate(cfg.quality, quality, meas_info)
            mm_per_px = scale_est.estimate_mm_per_px(frame, roi_offset=(ox, oy))
            gap_mm = float(gap_px) * float(mm_per_px)

        # Primary endpoints (ROI -> full)
        gap_x_left_roi = meas_info.get("gap_x_left_roi", None)
        gap_x_right_roi = meas_info.get("gap_x_right_roi", None)
        gap_x_left_full = (gap_x_left_roi + ox) if gap_x_left_roi is not None else None
        gap_x_right_full = (gap_x_right_roi + ox) if gap_x_right_roi is not None else None

        # Export ML labels only when depth mask exists (auto-label)
        export_labels = True
        if export_labels and mask01_depth is not None:
            label_root = "ml_dataset"
            img_dir = os.path.join(label_root, "images")
            msk_dir = os.path.join(label_root, "masks")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(msk_dir, exist_ok=True)

            safe_id = os.path.basename(str(frame_id))
            cv2.imwrite(os.path.join(img_dir, f"{safe_id}.png"), frame_roi)
            cv2.imwrite(os.path.join(msk_dir, f"{safe_id}.png"), (mask01_depth * 255).astype("uint8"))

        # Results row (full-frame coords first)
        row = {
            "frame_id": frame_id,
            "ok": bool(ok),
            "gap_px": float(gap_px) if gap_px is not None else None,
            "gap_mm": gap_mm,

            # PRIMARY endpoints (depth primary or seg fallback) - full then ROI
            "gap_x_left_full": gap_x_left_full,
            "gap_x_right_full": gap_x_right_full,
            "gap_x_left_roi": gap_x_left_roi,
            "gap_x_right_roi": gap_x_right_roi,

            # Depth endpoints + pipes (anchor)
            "depth_gap_x_left_full": depth_gap_x_left_full,
            "depth_gap_x_right_full": depth_gap_x_right_full,
            "gap_px_depth": gap_px_depth,
            "near_depth_mm": near_depth_mm,
            "pipe1_x0_full": pipe1_x0_full,
            "pipe1_x1_full": pipe1_x1_full,
            "pipe2_x0_full": pipe2_x0_full,
            "pipe2_x1_full": pipe2_x1_full,

            "roi_enabled": cfg.roi.enabled,
            "roi_x": ox,
            "roi_y": oy,

            **quality,
            **meas_info,
            **conf,
        }

        # ML secondary columns
        row.update({
            "ml_gap_px": ml_gap_px,
            "ml_gap_x_left_full": ml_gap_x_left_full,
            "ml_gap_x_right_full": ml_gap_x_right_full,
            "ml_gap_x_left_roi": ml_gap_x_left_roi,
            "ml_gap_x_right_roi": ml_gap_x_right_roi,
            "ml_valid_ratio": ml_valid_ratio,
            "ml_mask_coverage": ml_mask_coverage,
        })

        if gap_px_depth is not None and ml_gap_px is not None:
            row["ml_vs_depth_gap_px_abs_err"] = abs(float(ml_gap_px) - float(gap_px_depth))

        results.append(row)

        # Preview windows
        if cfg.debug.show_windows:
            cv2.imshow("ROI gray", gray)
            if mask01_used is not None:
                cv2.imshow("mask_primary_used", (mask01_used * 255).astype("uint8"))
            if ml_mask01 is not None:
                cv2.imshow("mask_ml", (ml_mask01 * 255).astype("uint8"))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        # Overlay: ML edges (solid) vs depth edges (dashed)
        if cfg.output.save_overlays and overlay_count < cfg.output.overlay_limit:
            save_overlay(
                out_dir=os.path.join(run_dir, "overlays"),
                frame_id=frame_id,
                frame_bgr=frame,
                roi_offset=(ox, oy),
                roi_size=roi_size,
                ok=bool(ok),
                overlay_index=overlay_count,

                ml_gap_x_left_full=ml_gap_x_left_full,
                ml_gap_x_right_full=ml_gap_x_right_full,
                ml_gap_px=ml_gap_px,

                depth_gap_x_left_full=depth_gap_x_left_full,
                depth_gap_x_right_full=depth_gap_x_right_full,
                depth_gap_px=gap_px_depth,

                pipe1_x0_full=pipe1_x0_full,
                pipe1_x1_full=pipe1_x1_full,
                pipe2_x0_full=pipe2_x0_full,
                pipe2_x1_full=pipe2_x1_full,
            )
            overlay_count += 1

    if cfg.debug.show_windows:
        cv2.destroyAllWindows()

    csv_path = write_results_csv(results, run_dir)
    summary_path = write_summary_json(results, run_dir)

    print(f"Saved results: {csv_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Overlays saved: {os.path.join(run_dir, 'overlays') if cfg.output.save_overlays else '(disabled)'}")


if __name__ == "__main__":
    main()
