import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RANGE_ORDER = ["near_0-10m", "middle_10-100m", "far_100-inf"]
RANGE_LABELS = {
    "near_0-10m": "Near field: 0-10 m",
    "middle_10-100m": "Middle field: 10-100 m",
    "far_100-inf": "Far field: >=100 m",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build a reproducible scene-level QC report from image-space RMSE and "
            "local depth-agreement errors."
        )
    )
    ap.add_argument(
        "--rmse-candidates",
        default="analysis/qc_review/low_reprojection_error_candidates_all_pools.csv",
        help="Scene-level table containing fit_rmse_total_px.",
    )
    ap.add_argument(
        "--local-depth-errors",
        default="analysis/qc_review/correspondence_distance_errors/per_correspondence_local_depth_errors.csv",
        help="Per-correspondence local depth disagreement table.",
    )
    ap.add_argument(
        "--out-dir",
        default="analysis/qc_review/reproducible_qc_report",
        help="Output directory for tables and plots.",
    )
    ap.add_argument(
        "--rmse-thresholds",
        default="2,3,5,10",
        help="Comma-separated image-space RMSE thresholds to report in pixels.",
    )
    ap.add_argument(
        "--distance-thresholds",
        default="1,5",
        help="Comma-separated local depth agreement thresholds to report in percent of range.",
    )
    return ap.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def scene_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["collection"], row["path"], row["step"]


def summarize_scene_depth(rows: list[dict[str, str]], thresholds: list[float]) -> dict[tuple[str, str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[scene_key(row)].append(row)

    summaries: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, scene_rows in grouped.items():
        sampled = [row for row in scene_rows if row.get("status") == "sampled"]
        missing = [row for row in scene_rows if row.get("status") == "missing_depth_near_pick"]
        scene_errors = [row for row in scene_rows if str(row.get("status", "")).startswith("scene_error")]
        percent_errors = np.asarray(
            [as_float(row.get("absolute_depth_error_percent_of_range", "")) for row in sampled],
            dtype=np.float64,
        )
        percent_errors = percent_errors[np.isfinite(percent_errors)]
        range_counts = Counter(row.get("distance_bin", "") for row in sampled)

        summary: dict[str, Any] = {
            "collection": key[0],
            "path": key[1],
            "step": key[2],
            "distance_sampled_points": int(len(percent_errors)),
            "distance_missing_points": int(len(missing)),
            "distance_total_correspondence_attempts": int(len(percent_errors) + len(missing)),
            "distance_scene_error_count": int(len(scene_errors)),
            "distance_has_evidence": bool(len(percent_errors) > 0),
            "distance_max_percent": float(np.max(percent_errors)) if len(percent_errors) else math.nan,
            "distance_p90_percent": float(np.percentile(percent_errors, 90)) if len(percent_errors) else math.nan,
            "distance_median_percent": float(np.median(percent_errors)) if len(percent_errors) else math.nan,
            "near_points": int(range_counts.get("near_0-10m", 0)),
            "middle_points": int(range_counts.get("middle_10-100m", 0)),
            "far_points": int(range_counts.get("far_100-inf", 0)),
        }
        for threshold in thresholds:
            name = f"distance_pass_all_points_le_{threshold:g}pct"
            summary[name] = bool(len(percent_errors) > 0 and len(missing) == 0 and len(scene_errors) == 0 and np.max(percent_errors) <= threshold)
            summary[f"distance_points_gt_{threshold:g}pct"] = int(np.sum(percent_errors > threshold))
            kept_after_drop = int(np.sum(percent_errors <= threshold))
            dropped_problematic = int(np.sum(percent_errors > threshold) + len(missing))
            summary[f"distance_points_kept_after_drop_gt_{threshold:g}pct"] = kept_after_drop
            summary[f"distance_points_dropped_gt_{threshold:g}pct_or_missing"] = dropped_problematic
            summary[f"distance_pass_after_drop_gt_{threshold:g}pct_min8"] = bool(
                len(scene_errors) == 0
                and (len(percent_errors) + len(missing)) >= 9
                and kept_after_drop >= 8
            )
        summaries[key] = summary
    return summaries


def build_scene_table(
    rmse_rows: list[dict[str, str]],
    depth_rows: list[dict[str, str]],
    rmse_thresholds: list[float],
    distance_thresholds: list[float],
) -> list[dict[str, Any]]:
    depth_summary = summarize_scene_depth(depth_rows, distance_thresholds)
    rows: list[dict[str, Any]] = []
    for row in rmse_rows:
        key = scene_key(row)
        fit_rmse = as_float(row.get("fit_rmse_total_px", ""))
        calibration_source = (
            "own_fitted_cyl_from_picked_correspondences"
            if row.get("source") == "annotation_workspace_nocyl"
            else "inherited_scene_cyl_plus_own_rigid_lidar_fit"
        )
        out: dict[str, Any] = {
            "collection": row["collection"],
            "path": row["path"],
            "step": row["step"],
            "scene": row.get("title") or f"{row['collection']} / {row['path']} / {row['step']}",
            "source": row.get("source", ""),
            "calibration_source": calibration_source,
            "fit_rmse_total_px": fit_rmse,
            "num_picked_pairs": row.get("num_picked_pairs", ""),
            "disk_overlay": row.get("disk_overlay", ""),
            "disk_reference": row.get("disk_reference", ""),
            "fit_path": row.get("fit_path", ""),
        }
        for threshold in rmse_thresholds:
            out[f"rmse_pass_le_{threshold:g}px"] = bool(np.isfinite(fit_rmse) and fit_rmse <= threshold)
        out.update(depth_summary.get(key, {}))
        if key not in depth_summary:
            out.update(
                {
                    "distance_sampled_points": 0,
                    "distance_missing_points": 0,
                    "distance_total_correspondence_attempts": 0,
                    "distance_scene_error_count": 0,
                    "distance_has_evidence": False,
                }
            )
            for threshold in distance_thresholds:
                out[f"distance_pass_all_points_le_{threshold:g}pct"] = False
                out[f"distance_points_gt_{threshold:g}pct"] = ""
                out[f"distance_points_kept_after_drop_gt_{threshold:g}pct"] = ""
                out[f"distance_points_dropped_gt_{threshold:g}pct_or_missing"] = ""
                out[f"distance_pass_after_drop_gt_{threshold:g}pct_min8"] = False
        for threshold in distance_thresholds:
            out[f"candidate_include_rmse10_distance{threshold:g}pct"] = bool(
                out.get("rmse_pass_le_10px") and out.get(f"distance_pass_all_points_le_{threshold:g}pct")
            )
        rows.append(out)
    return rows


def make_threshold_counts(scene_rows: list[dict[str, Any]], distance_thresholds: list[float]) -> list[dict[str, Any]]:
    total = len(scene_rows)
    rows: list[dict[str, Any]] = []
    for threshold in distance_thresholds:
        pass_key = f"distance_pass_all_points_le_{threshold:g}pct"
        relaxed_key = f"distance_pass_after_drop_gt_{threshold:g}pct_min8"
        rmse_distance_key = f"candidate_include_rmse10_distance{threshold:g}pct"
        pass_distance = sum(1 for row in scene_rows if row.get(pass_key) is True)
        pass_relaxed = sum(1 for row in scene_rows if row.get(relaxed_key) is True)
        recovered = sum(
            1
            for row in scene_rows
            if row.get(pass_key) is not True and row.get(relaxed_key) is True
        )
        pass_with_drop_rule = pass_distance + recovered
        pass_both = sum(1 for row in scene_rows if row.get(rmse_distance_key) is True)
        rows.append(
            {
                "rule": f"distance_all_points <= {threshold:g}% of range",
                "scenes_pass": pass_distance,
                "scenes_fail_or_missing": total - pass_distance,
                "scenes_lost": total - pass_distance,
                "scenes_pass_after_dropping_problem_points_min8": pass_relaxed,
                "scenes_accepted_with_drop_rule": pass_with_drop_rule,
                "scenes_recovered_by_drop_rule": recovered,
                "scenes_fail_with_drop_rule": total - pass_with_drop_rule,
                "scenes_pass_with_rmse_le_10px": pass_both,
                "scenes_lost_with_rmse_le_10px": total - pass_both,
                "total_scenes": total,
            }
        )
    return rows


def make_rmse_counts(scene_rows: list[dict[str, Any]], rmse_thresholds: list[float]) -> list[dict[str, Any]]:
    total = len(scene_rows)
    rows: list[dict[str, Any]] = []
    values = np.asarray([row["fit_rmse_total_px"] for row in scene_rows], dtype=np.float64)
    finite = values[np.isfinite(values)]
    for threshold in rmse_thresholds:
        pass_count = sum(1 for row in scene_rows if row.get(f"rmse_pass_le_{threshold:g}px") is True)
        rows.append(
            {
                "rmse_threshold_px": threshold,
                "scenes_pass": pass_count,
                "scenes_lost": total - pass_count,
                "total_scenes": total,
            }
        )
    rows.append(
        {
            "rmse_threshold_px": "distribution",
            "scenes_pass": "",
            "scenes_lost": "",
            "total_scenes": total,
            "rmse_median_px": float(np.median(finite)) if len(finite) else math.nan,
            "rmse_p75_px": float(np.percentile(finite, 75)) if len(finite) else math.nan,
            "rmse_p90_px": float(np.percentile(finite, 90)) if len(finite) else math.nan,
            "rmse_p95_px": float(np.percentile(finite, 95)) if len(finite) else math.nan,
            "rmse_max_px": float(np.max(finite)) if len(finite) else math.nan,
        }
    )
    return rows


def distance_percent_values(depth_rows: list[dict[str, str]], range_key: str) -> np.ndarray:
    sampled = [row for row in depth_rows if row.get("status") == "sampled"]
    values = np.asarray(
        [
            as_float(row.get("absolute_depth_error_percent_of_range", ""))
            for row in sampled
            if row.get("distance_bin") == range_key
        ],
        dtype=np.float64,
    )
    return values[np.isfinite(values)]


def plot_distance_percent_histograms(depth_rows: list[dict[str, str]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    bins = np.concatenate([np.linspace(0, 10, 41), np.asarray([15, 25, 50, 100, 200])])
    for ax, range_key in zip(axes, RANGE_ORDER):
        values = distance_percent_values(depth_rows, range_key)
        clipped = np.clip(values, 0, 200)
        ax.hist(clipped, bins=bins, color="#1f6f78", edgecolor="white", linewidth=0.45)
        ax.axvline(1, color="#2f8f46", linewidth=2.0, label="1% target")
        ax.axvline(5, color="#b7372f", linewidth=2.0, label="5% reject")
        ax.set_title(f"{RANGE_LABELS[range_key]}\nn={len(values)}")
        ax.set_xlabel("Local depth error (% of picked range)")
        ax.grid(axis="y", alpha=0.25)
        ax.set_xlim(0, 25)
    axes[0].set_ylabel("Picked correspondences")
    axes[-1].legend(frameon=False, loc="upper right")
    fig.suptitle("Distance Agreement: Picked 3D Range vs. Local Projected Depth Label", y=1.08)
    fig.text(0.5, -0.03, "Histogram bin width is 0.25 percentage points from 0-10%; long-tail values are shown up to 25%.", ha="center")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_single_range_distance_histograms(depth_rows: list[dict[str, str]], out_dir: Path) -> None:
    bins = np.concatenate([np.linspace(0, 10, 41), np.asarray([15, 25, 50, 100, 200])])
    for range_key in RANGE_ORDER:
        values = distance_percent_values(depth_rows, range_key)
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.hist(np.clip(values, 0, 200), bins=bins, color="#1f6f78", edgecolor="white", linewidth=0.45)
        ax.axvspan(0, 1, color="#2f8f46", alpha=0.12, label="<=1% target")
        ax.axvline(1, color="#2f8f46", linewidth=2.2)
        ax.axvline(5, color="#b7372f", linewidth=2.2, label="5% reject")
        ax.set_xlim(0, 25)
        ax.set_xlabel("Local depth error (% of picked 3D range)")
        ax.set_ylabel("Picked correspondences")
        ax.set_title(f"{RANGE_LABELS[range_key]} distance agreement (n={len(values)})")
        ax.text(
            0.98,
            0.86,
            "0.25 percentage-point bins\nvalues >25% omitted from view",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"distance_error_percent_{range_key}.png", dpi=220)
        plt.close(fig)


def plot_scene_pass_counts(counts: list[dict[str, Any]], out_path: Path) -> None:
    labels = [row["rule"].replace("distance_all_points ", "") for row in counts]
    passed = np.asarray([row["scenes_pass"] for row in counts], dtype=np.float64)
    lost = np.asarray([row["scenes_lost"] for row in counts], dtype=np.float64)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(x, passed, color="#2f8f46", label="Pass")
    ax.bar(x, lost, bottom=passed, color="#b7372f", label="Fail or missing")
    for idx, (p, l) in enumerate(zip(passed, lost)):
        ax.text(idx, p / 2, f"{int(p)}", ha="center", va="center", color="white", fontweight="bold")
        ax.text(idx, p + l / 2, f"{int(l)}", ha="center", va="center", color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Scenes")
    ax.set_title("Scene Retention Under Distance-Agreement Rules")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_rmse_distribution(scene_rows: list[dict[str, Any]], out_path: Path) -> None:
    values = np.asarray([float(row["fit_rmse_total_px"]) for row in scene_rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(values, bins=36, color="#324f7c", edgecolor="white", linewidth=0.45)
    for threshold, color in [(3, "#2f8f46"), (5, "#d08c28"), (10, "#b7372f")]:
        ax.axvline(threshold, color=color, linewidth=2.0, label=f"{threshold}px")
    ax.set_xlabel("Image-space fit RMSE (pixels)")
    ax.set_ylabel("Scenes")
    ax.set_title("Image-Space Registration Error Distribution")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    rmse_thresholds = [float(v) for v in args.rmse_thresholds.split(",")]
    distance_thresholds = [float(v) for v in args.distance_thresholds.split(",")]
    rmse_rows = read_csv(Path(args.rmse_candidates))
    depth_rows = read_csv(Path(args.local_depth_errors))

    scene_rows = build_scene_table(rmse_rows, depth_rows, rmse_thresholds, distance_thresholds)
    distance_counts = make_threshold_counts(scene_rows, distance_thresholds)
    rmse_counts = make_rmse_counts(scene_rows, rmse_thresholds)

    write_csv(out_dir / "scene_reproducible_qc_summary.csv", scene_rows)
    write_csv(out_dir / "distance_threshold_scene_counts.csv", distance_counts)
    write_csv(out_dir / "rmse_threshold_scene_counts.csv", rmse_counts)
    for threshold in distance_thresholds:
        failing = [
            row
            for row in scene_rows
            if row.get(f"distance_pass_all_points_le_{threshold:g}pct") is not True
        ]
        failing.sort(
            key=lambda row: (
                -float(row.get("distance_max_percent") or -1)
                if np.isfinite(float(row.get("distance_max_percent") or math.nan))
                else 1,
                row["scene"],
            )
        )
        write_csv(out_dir / f"scenes_failing_distance_{threshold:g}pct.csv", failing)
        recovered = [
            row
            for row in scene_rows
            if row.get(f"distance_pass_all_points_le_{threshold:g}pct") is not True
            and row.get(f"distance_pass_after_drop_gt_{threshold:g}pct_min8") is True
        ]
        write_csv(out_dir / f"scenes_recovered_by_dropping_distance_{threshold:g}pct.csv", recovered)
        still_failing_after_drop = [
            row
            for row in scene_rows
            if row.get(f"distance_pass_all_points_le_{threshold:g}pct") is not True
            and row.get(f"distance_pass_after_drop_gt_{threshold:g}pct_min8") is not True
        ]
        write_csv(out_dir / f"scenes_still_failing_after_dropping_distance_{threshold:g}pct.csv", still_failing_after_drop)

    plot_distance_percent_histograms(depth_rows, out_dir / "distance_error_percent_by_range.png")
    plot_single_range_distance_histograms(depth_rows, out_dir)
    plot_scene_pass_counts(distance_counts, out_dir / "distance_threshold_scene_retention.png")
    plot_rmse_distribution(scene_rows, out_dir / "rmse_distribution.png")

    print(f"Scenes: {len(scene_rows)}")
    print(f"Scene summary: {out_dir / 'scene_reproducible_qc_summary.csv'}")
    print(f"Distance counts: {out_dir / 'distance_threshold_scene_counts.csv'}")
    print(f"RMSE counts: {out_dir / 'rmse_threshold_scene_counts.csv'}")
    print(f"Plots: {out_dir}")


if __name__ == "__main__":
    main()
