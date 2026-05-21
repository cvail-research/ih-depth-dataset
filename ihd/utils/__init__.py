from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "DEPTH_SCALE": "ihd.utils.depth_png",
    "DepthEvalConfig": "ihd.utils.depth_metrics",
    "build_prediction_input_rows_from_scene_manifest": "ihd.utils.baseline_io",
    "decode_depth_u16": "ihd.utils.depth_png",
    "depth_metrics_from_arrays": "ihd.utils.depth_metrics",
    "encode_depth_u16": "ihd.utils.depth_png",
    "evaluate_scene_pair": "ihd.utils.evaluation",
    "format_summary_text": "ihd.utils.evaluation",
    "infer_sensor_metadata": "ihd.utils.baseline_io",
    "load_depth_png": "ihd.utils.depth_png",
    "load_ground_truth_depth": "ihd.utils.baseline_io",
    "load_hyperspectral_cube": "ihd.utils.baseline_io",
    "load_pseudobroadband_rgb": "ihd.utils.baseline_io",
    "pair_evaluation_files": "ihd.utils.evaluation",
    "read_prediction_input_manifest": "ihd.utils.baseline_io",
    "save_depth_png": "ihd.utils.depth_png",
    "save_depth_prediction": "ihd.utils.baseline_io",
    "save_input_prediction_groundtruth_figures": "ihd.utils.baseline_io",
    "scene_out_dir": "ihd.utils.baseline_io",
    "summarize_metric_rows": "ihd.utils.depth_metrics",
    "write_prediction_manifest": "ihd.utils.baseline_io",
    "write_stats_report": "ihd.utils.evaluation",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
