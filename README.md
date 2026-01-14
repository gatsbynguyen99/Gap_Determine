Overview

This project estimates the gap size between two pipes from images. It supports:

Offline “captures” mode: reads synchronized RGB, depth (mm), and meta JSON from a dataset folder.

Live camera mode: reads frames from a USB camera (RGB only by default).

The system is designed as a modular pipeline:

ROI selection

preprocessing

segmentation (multiple methods)

gap measurement (scanlines)

confidence/quality gating

scaling (mm/pixel)

reporting + overlays

Repository Structure (key files)
configs/
  default.json              # runtime config

mockup_dataset/
  captures/                 # rgb_<ts>.png, depth_mm_<ts>.png, meta_<ts>.json

src/
  main.py                   # pipeline orchestrator
  pipeline/
    config.py               # typed config loader
    io_sources.py           # folder/camera/captures iterators
    roi.py                  # manual ROI cropping
    preprocess.py           # grayscale + CLAHE + denoise
    segment.py              # segmentation methods + depth logic + component filtering
    measure.py              # scanline measurement utilities
    quality.py              # blur/saturation metrics + gating
    scale.py                # mm_per_px stub (fixed now)
    report.py               # CSV + summary JSON
    viz.py                  # overlay images

Data / Dataset Format (Offline Captures)

Offline captures are expected in a single folder with matching timestamps:

rgb_<ts>.png — RGB image

depth_mm_<ts>.png — depth image in millimeters (uint16), invalid=0

meta_<ts>.json — metadata

Example:

mockup_dataset/captures/
  rgb_20260113_024652.png
  depth_mm_20260113_024652.png
  meta_20260113_024652.json

Pipeline: Stage-by-Stage Explanation
1) Source Input (pipeline/io_sources.py)

folder: loads images using a glob pattern.

camera: streams from a camera index via OpenCV.

captures: loads RGB+depth+meta by timestamp.

Output per frame:

RGB frame (frame_bgr)

optional depth frame (depth_mm)

frame identifier (frame_id/timestamp)

2) ROI Cropping (pipeline/roi.py)

Manual ROI is used to reduce false edges and background clutter:

ROI defined as (x, y, w, h) in configs/default.json

returns frame_roi and (offset_x, offset_y) for overlays

Why ROI matters: background pipes and scene edges can dominate gradients and masks if ROI is too wide.

3) Preprocessing (pipeline/preprocess.py)

Two grayscale streams are used:

gray_raw: raw grayscale (no CLAHE/blur) — used for blur/quality metric

gray: preprocessed grayscale — used for segmentation/measurement

Preprocess includes:

grayscale conversion

CLAHE (local contrast normalization)

light Gaussian blur (noise reduction)

4) Segmentation (pipeline/segment.py)

Segmentation produces a binary mask mask01 (0/1) for the region used to compute gap width.

Implemented methods:

Otsu (global threshold): fast baseline; works only if ROI histogram is bimodal.

Adaptive threshold: local threshold; better under uneven illumination but still intensity-dependent.

Edge-guided (top-K): uses gradients to find left/right boundaries per scanline and fills between them; best classical method so far.

ML ONNX scaffold: code supports inference via OpenCV DNN (cv2.dnn.readNetFromONNX), but requires a trained .onnx model file.

Background-pipes mitigation (Layer B)

After segmentation, connected-component filtering selects the most plausible component:

prefers tall components (reject short horizontal artifacts)

favors reasonable bbox width (expected gap range)

optionally uses depth as a preference signal (closer/farther logic depends on scene)

Depth usage (current)

Depth can be used for additional filtering/scoring.

Note: in some scenes the “gap region” is actually background wall (farther), so depth gating must be applied carefully (often better for Otsu/adaptive than edge-guided).

5) Gap Measurement (pipeline/measure.py / segment.py)

Gap width is computed using scanlines:

choose many horizontal rows between scanline_y_min and scanline_y_max

for each row, find the longest contiguous run of mask01==1

aggregate width across scanlines using median → gap_px

Why median:

robust to outlier rows caused by noise, shadows, or partial mask failures

6) Quality / Confidence Gate (pipeline/quality.py)

Per-frame quality checks include:

blur metric: variance of Laplacian computed on gray_raw

saturation ratio: fraction of pixels near 0 or 255

valid_ratio: fraction of scanlines producing usable width

edge_strength: placeholder for mask-based measurement (999.0)

Frame output is accepted only if thresholds pass, e.g.:

blur_var >= blur_min_var

valid_ratio >= min_valid_ratio

7) Scaling (pixels → mm) (pipeline/scale.py)

Currently uses a fixed config value:

gap_mm = gap_px * mm_per_px

Current state: mm_per_px is a placeholder.
For the mock setup: pipes are ~23 cm from camera, true gap is ~10 mm. A simple calibration is:

mm_per_px ≈ 10 / gap_px

Future: depth + intrinsics (fx) can provide approximate scaling:

mm_per_px ≈ Z / fx

8) Outputs (pipeline/report.py, pipeline/viz.py)

Each run creates:

outputs/<timestamp>/results.csv

outputs/<timestamp>/summary.json

outputs/<timestamp>/overlays/overlay_####.jpg (ROI rectangle + id + gap)

outputs/<timestamp>/masks/mask_<id>.png (binary masks) — used for debugging segmentation

How to Run
1) Install dependencies
pip install opencv-python numpy pandas

2) Validate config JSON
python -m json.tool configs/default.json

3) Offline captures (recommended for development)

Set in configs/default.json:

"source": { "type": "captures", "path": "mockup_dataset/captures", "max_frames": 0 }


Run:

python src/main.py --config configs/default.json

4) Live camera

Set:

"source": { "type": "camera", "camera_index": 0, "max_frames": 300 }


Run:

python src/main.py --config configs/default.json


Press q to quit preview.

What We’ve Observed So Far (Status)
Otsu / Adaptive

Often produce sparse or incorrect masks (thin horizontal artifacts or empty masks) in this scene.

This causes:

no_gap_found

very low valid_ratio

tiny gap_px values not matching the physical gap

Edge-guided (top-K)

Produces more plausible gap estimates in pixels (closer to expected).

Remaining issue: some scanlines fail to find good boundary pairs, resulting in “horizontal bars” masks and low valid_ratio.

Next tuning directions:

make gap range constraints “soft” (penalty instead of hard rejection)

increase edge_top_k

tighten ROI and scanline vertical range

tighten expected gap range in pixels once stable

ML (ONNX) plan

Pipeline includes ONNX inference scaffolding using OpenCV DNN.

Next steps for ML:

generate pseudo-label masks from best classical method

manually correct 30–100 masks

train U-Net-like segmentation model

export to .onnx

run with segmentation.method = "ml_onnx"

Configuration Tips (Most Important)

ROI is critical when background pipes exist.

expected_gap_px_min/max should be tightened after you see stable edge-guided results.

Save masks to debug segmentation:

inspect outputs/<run>/masks/ to see what the algorithm is actually segmenting.

Next Work Items

Improve edge-guided stability:

soft gap-range penalty instead of hard thresholds

increase top-K candidates

restrict scanline band to stable region

Calibrate mm_per_px using known 10 mm gap (or compute per-run median)

Begin labeling workflow for ML segmentation

Integrate depth more safely (depending on whether gap corresponds to background depth)

Notes

ONNX is the model file format; OpenCV DNN is the runtime that executes it.

Depth is in millimeters and aligned to RGB in offline captures.