import argparse
import json
from pathlib import Path


ACCEPTED_VERDICTS = {"good", "usable with caution"}


def parse_args():
    ap = argparse.ArgumentParser(
        description="Aggregate per-scene lidar labeling summaries into a compact evaluation report."
    )
    ap.add_argument("--results-root", required=True, help="Root directory containing per-scene result folders")
    ap.add_argument("--scene", action="append", required=True, help="Scene folder name under results-root")
    ap.add_argument("--out", required=True, help="Output text report path")
    return ap.parse_args()


def read_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def mean(values):
    return sum(values) / len(values) if values else 0.0


def main():
    args = parse_args()
    root = Path(args.results_root)
    scenes = []
    for scene_name in args.scene:
        summary_json = root / scene_name / "summary.json"
        data = read_summary(summary_json)
        data["scene_name"] = scene_name
        scenes.append(data)

    accepted = [s for s in scenes if s.get("verdict", "").strip() in ACCEPTED_VERDICTS]
    rejected = [s for s in scenes if s.get("verdict", "").strip() not in ACCEPTED_VERDICTS]

    ann_all = [float(s["annotation_minutes"]) for s in scenes]
    ann_acc = [float(s["annotation_minutes"]) for s in accepted]
    proc_all = [float(s["processing_minutes"]) for s in scenes]
    total_all = [float(s["total_minutes"]) for s in scenes]
    rmse_all = [float(s["fit_rmse_total"]) for s in scenes]
    rmse_acc = [float(s["fit_rmse_total"]) for s in accepted]

    hours_60_all = mean(total_all) * 60.0 / 60.0
    hours_60_acc = mean(ann_acc) * 60.0 / 60.0 if ann_acc else 0.0

    lines = [
        f"scene_count: {len(scenes)}",
        f"accepted_count: {len(accepted)}",
        f"rejected_count: {len(rejected)}",
        f"acceptance_rate: {len(accepted) / len(scenes):.6f}" if scenes else "acceptance_rate: 0.000000",
        f"avg_annotation_minutes_all: {mean(ann_all):.6f}",
        f"avg_annotation_minutes_accepted: {mean(ann_acc):.6f}",
        f"avg_processing_minutes_all: {mean(proc_all):.6f}",
        f"avg_total_minutes_all: {mean(total_all):.6f}",
        f"avg_fit_rmse_total_all: {mean(rmse_all):.6f}",
        f"avg_fit_rmse_total_accepted: {mean(rmse_acc):.6f}",
        f"projected_hours_for_60_scenes_all: {hours_60_all:.6f}",
        f"projected_hours_for_60_scenes_accepted_annotation_only: {hours_60_acc:.6f}",
        "scene_table:",
    ]
    for s in scenes:
        lines.append(
            "  "
            f"{s['scene_name']}: verdict={s.get('verdict','')}, "
            f"annotation_minutes={float(s['annotation_minutes']):.6f}, "
            f"fit_rmse_total={float(s['fit_rmse_total']):.6f}, "
            f"cyl_verify_rmse_total={float(s['cyl_verify_rmse_total']):.6f}"
        )

    Path(args.out).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
