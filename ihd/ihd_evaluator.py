from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ihd.utils.depth_metrics import DepthEvalConfig
from ihd.utils.evaluation import evaluate_scene_pair, pair_evaluation_files, write_stats_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate IH-Depth uint16 PNG predictions against mirrored GT PNGs. "
            "Depth encoding is stored_value = round(128 * depth_m) with 0 reserved for invalid pixels."
        )
    )
    parser.add_argument("gt_dir", metavar="GT_DIR")
    parser.add_argument("prediction_dir", metavar="PREDICTION_DIR")
    parser.add_argument("--output_name", default="stats_ihd.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_pairs = pair_evaluation_files(args.gt_dir, args.prediction_dir)
    rows = [
        evaluate_scene_pair(relative_path, gt_path, pred_path, args.prediction_dir, DepthEvalConfig())
        for relative_path, gt_path, pred_path in scene_pairs
    ]
    _report_path, _summary, text = write_stats_report(args.prediction_dir, args.output_name, rows)
    print(text, end="")


if __name__ == "__main__":
    main()
