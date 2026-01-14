import json
from dataclasses import dataclass
from typing import Any, Dict, Tuple


# -------------------------
# Source configuration
# -------------------------
@dataclass(frozen=True)
class SourceCfg:
    type: str                 # "folder" or "camera"
    path: str                 # used when folder
    glob: str                 # used when folder
    max_frames: int
    camera_index: int         # used when camera


# -------------------------
# ROI configuration
# -------------------------
@dataclass(frozen=True)
class RoiCfg:
    enabled: bool
    x: int
    y: int
    w: int
    h: int


# -------------------------
# Preprocess configuration
# -------------------------
@dataclass(frozen=True)
class PreprocessCfg:
    clahe_clip: float
    clahe_grid: Tuple[int, int]
    blur_ksize: int


# -------------------------
# Measurement configuration
# -------------------------
@dataclass(frozen=True)
class MeasurementCfg:
    method: str
    n_scanlines: int
    scanline_y_min: float
    scanline_y_max: float
    min_gap_px: int


# -------------------------
# Quality configuration
# -------------------------
@dataclass(frozen=True)
class QualityCfg:
    blur_min_var: float
    max_saturation_ratio: float
    min_valid_ratio: float
    min_edge_strength: float


# -------------------------
# Scale configuration
# -------------------------
@dataclass(frozen=True)
class ScaleCfg:
    mode: str
    mm_per_px: float


# -------------------------
# Segmentation configuration (ONE definition only)
# -------------------------
@dataclass(frozen=True)
class SegmentationCfg:
    enabled: bool
    method: str  # "otsu" | "adaptive" | "edge_guided" | (later "ml_onnx")

    # Layer B component filter
    component_filter_enabled: bool
    expected_gap_px_min: int
    expected_gap_px_max: int

    # Option 1 adaptive thresholding
    adaptive_block_size: int
    adaptive_C: int

    # Option 2 edge-guided top-K
    edge_top_k: int

    # Layer D temporal prior (camera stream)
    use_temporal_prior: bool

    #ML ONNX fiels
    model_path: str
    onnx_input_size: Tuple[int, int]
    onnx_thresh: float



# -------------------------
# Output configuration
# -------------------------
@dataclass(frozen=True)
class OutputCfg:
    dir: str
    save_overlays: bool
    overlay_limit: int


# -------------------------
# Debug configuration
# -------------------------
@dataclass(frozen=True)
class DebugCfg:
    show_windows: bool


# -------------------------
# App configuration root
# -------------------------
@dataclass(frozen=True)
class AppCfg:
    source: SourceCfg
    roi: RoiCfg
    preprocess: PreprocessCfg
    measurement: MeasurementCfg
    quality: QualityCfg
    scale: ScaleCfg
    segmentation: SegmentationCfg
    output: OutputCfg
    debug: DebugCfg


def _require(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise ValueError(f"Missing required config key: {key}")
    return d[key]


def load_config(path: str) -> AppCfg:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    src = _require(raw, "source")
    roi = _require(raw, "roi")
    pre = _require(raw, "preprocess")
    meas = _require(raw, "measurement")
    qual = _require(raw, "quality")
    scale = _require(raw, "scale")
    seg = _require(raw, "segmentation")
    out = _require(raw, "output")
    dbg = _require(raw, "debug")

    source_cfg = SourceCfg(
        type=src.get("type", "folder"),
        path=src.get("path", "data/images"),
        glob=src.get("glob", "*.jpg"),
        max_frames=int(src.get("max_frames", 0)),
        camera_index=int(src.get("camera_index", 0)),
    )

    seg_cfg = SegmentationCfg(
        enabled=bool(seg.get("enabled", False)),
        method=str(seg.get("method", "otsu")),
        component_filter_enabled=bool(seg.get("component_filter_enabled", True)),
        expected_gap_px_min=int(seg.get("expected_gap_px_min", 5)),
        expected_gap_px_max=int(seg.get("expected_gap_px_max", 300)),
        adaptive_block_size=int(seg.get("adaptive_block_size", 31)),
        adaptive_C=int(seg.get("adaptive_C", 3)),
        edge_top_k=int(seg.get("edge_top_k", 3)),
        use_temporal_prior=bool(seg.get("use_temporal_prior", False)),

        # NEW
        model_path=str(seg.get("model_path", "")),
        onnx_input_size=tuple(seg.get("onnx_input_size", [256, 256])),
        onnx_thresh=float(seg.get("onnx_thresh", 0.5)),
    )

    return AppCfg(
        source=source_cfg,
        roi=RoiCfg(
            enabled=bool(roi.get("enabled", False)),
            x=int(roi.get("x", 0)),
            y=int(roi.get("y", 0)),
            w=int(roi.get("w", 0)),
            h=int(roi.get("h", 0)),
        ),
        preprocess=PreprocessCfg(
            clahe_clip=float(pre.get("clahe_clip", 2.0)),
            clahe_grid=tuple(pre.get("clahe_grid", [8, 8])),
            blur_ksize=int(pre.get("blur_ksize", 3)),
        ),
        measurement=MeasurementCfg(
            method=str(meas.get("method", "edges_scanlines")),
            n_scanlines=int(meas.get("n_scanlines", 80)),
            scanline_y_min=float(meas.get("scanline_y_min", 0.15)),
            scanline_y_max=float(meas.get("scanline_y_max", 0.85)),
            min_gap_px=int(meas.get("min_gap_px", 2)),
        ),
        quality=QualityCfg(
            blur_min_var=float(qual.get("blur_min_var", 60.0)),
            max_saturation_ratio=float(qual.get("max_saturation_ratio", 0.03)),
            min_valid_ratio=float(qual.get("min_valid_ratio", 0.7)),
            min_edge_strength=float(qual.get("min_edge_strength", 10.0)),
        ),
        scale=ScaleCfg(
            mode=str(scale.get("mode", "fixed")),
            mm_per_px=float(scale.get("mm_per_px", 0.05)),
        ),
        segmentation=seg_cfg,
        output=OutputCfg(
            dir=str(out.get("dir", "outputs")),
            save_overlays=bool(out.get("save_overlays", True)),
            overlay_limit=int(out.get("overlay_limit", 30)),
        ),
        debug=DebugCfg(
            show_windows=bool(dbg.get("show_windows", False)),
        ),
    )
