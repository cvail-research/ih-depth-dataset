from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ihd.utils.baseline_io import (
    build_prediction_input_rows_from_scene_manifest,
    infer_sensor_metadata,
    load_ground_truth_depth,
    load_pseudobroadband_rgb,
    read_prediction_input_manifest,
    save_depth_prediction,
    save_input_prediction_groundtruth_figures,
    scene_out_dir,
    write_prediction_manifest,
)
from baselines.physics_based.run_quadspectral import _pick_bands, quadspectral_distance
from baselines.physics_based.utils.io_utils import load_scene
from baselines.physics_based.utils.physics import estimate_T_air


MODEL_SLUG = "quadspectral"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run physics-based quadspectral depth inference.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hdr", help="Single ENVI .hdr path.")
    src.add_argument("--manifest", help="CSV with hdr_path and optional label_path columns.")
    src.add_argument("--scene-manifest", help="Scene manifest with collection/path/step columns.")
    ap.add_argument("--label-path", help="Ground-truth IH-Depth uint16 PNG path when using --hdr.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-dir", default="baselines/physics_based/data")
    ap.add_argument(
        "--attenuation-profile",
        choices=["auto", "standard", "ozone_cues"],
        default="auto",
        help="Attenuation source profile.",
    )
    ap.add_argument(
        "--depth-label-root",
        help="Optional root for resolving public IH-Depth PNG labels; defaults to --disk-root.",
    )
    ap.add_argument("--disk-root", default="/disk")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--t-air", type=float, default=None)
    ap.add_argument("--lambda-min", type=float, default=8.5)
    ap.add_argument("--lambda-max", type=float, default=12.0)
    ap.add_argument("--cor-coeff", type=float, default=1.0)
    ap.add_argument("--idx1", type=int, default=None)
    ap.add_argument("--idx2", type=int, default=None)
    ap.add_argument("--idx3", type=int, default=None)
    ap.add_argument("--idx4", type=int, default=None)
    ap.add_argument("--sensor-id", choices=["LWHSI1", "LWHSI2"], default=None, help="Optional sensor filter.")
    ap.add_argument("--no-vis", action="store_true")
    return ap.parse_args()


def predict_one(
    hdr_path: str,
    out_dir: Path,
    data_dir: Path,
    t_air: float | None,
    lambda_min: float,
    lambda_max: float,
    cor_coeff: float,
    idx1: int | None,
    idx2: int | None,
    idx3: int | None,
    idx4: int | None,
    save_vis: bool,
    label_path: str | None = None,
    attenuation_profile: str = "auto",
) -> Path:
    meas, lambda_um, attenuation, _downwelling, sensor = load_scene(
        hdr_path,
        str(data_dir / "ozone_cues"),
        str(data_dir / "standard"),
        attenuation_profile=attenuation_profile,
    )
    if t_air is None:
        t_air_est, _ = estimate_T_air(meas, lambda_um, attenuation, lambda_min=lambda_min, lambda_max=lambda_max)
        t_air = float(t_air_est)
    if None in (idx1, idx2, idx3, idx4):
        idx1, idx2, idx3, idx4 = _pick_bands(lambda_um, attenuation, lambda_min, lambda_max)

    d_hat = quadspectral_distance(
        lambda_um,
        meas,
        attenuation,
        int(idx1),
        int(idx2),
        int(idx3),
        int(idx4),
        float(t_air),
        float(cor_coeff),
    )
    method = f"quadspectral_{sensor}"
    metadata = {
        "model_slug": MODEL_SLUG,
        "method_name": method,
        "sensor": sensor,
        "attenuation_profile": attenuation_profile,
        "t_air_k": float(t_air),
        "cor_coeff": float(cor_coeff),
        "idx1": int(idx1),
        "idx2": int(idx2),
        "idx3": int(idx3),
        "idx4": int(idx4),
        "lambda_idx1_um": float(lambda_um[int(idx1)]),
        "lambda_idx2_um": float(lambda_um[int(idx2)]),
        "lambda_idx3_um": float(lambda_um[int(idx3)]),
        "lambda_idx4_um": float(lambda_um[int(idx4)]),
    }
    pred_path = save_depth_prediction(
        d_hat,
        out_dir,
        model_name=method,
        hdr_path=hdr_path,
        metadata=metadata,
        save_visualization=save_vis,
    )
    rgb, _ = load_pseudobroadband_rgb(hdr_path)
    input_gray_u8 = np.mean(rgb.astype(np.float32), axis=2).clip(0, 255).astype(np.uint8)
    gt_depth = None
    gt_mask = None
    if label_path and Path(label_path).exists():
        gt_depth, gt_mask = load_ground_truth_depth(label_path)

    save_input_prediction_groundtruth_figures(
        input_gray_u8=input_gray_u8,
        prediction_m=np.asarray(d_hat, dtype=np.float32),
        out_dir=out_dir,
        ground_truth_m=gt_depth,
        ground_truth_mask=gt_mask,
    )
    return pred_path


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    rows_out = []
    if args.hdr:
        if args.sensor_id:
            sid, _ = infer_sensor_metadata(args.hdr)
            if sid != args.sensor_id:
                raise SystemExit(f"--hdr sensor mismatch: requested {args.sensor_id}, got {sid!r} for {args.hdr}")
        pred = predict_one(
            args.hdr,
            Path(args.out_dir),
            data_dir,
            args.t_air,
            args.lambda_min,
            args.lambda_max,
            args.cor_coeff,
            args.idx1,
            args.idx2,
            args.idx3,
            args.idx4,
            not args.no_vis,
            args.label_path,
            args.attenuation_profile,
        )
        print(pred)
        return

    if args.scene_manifest:
        rows_in = build_prediction_input_rows_from_scene_manifest(
            args.scene_manifest,
            depth_label_root=args.depth_label_root,
            disk_root=args.disk_root,
            limit=args.limit,
        )
    else:
        rows_in = read_prediction_input_manifest(args.manifest)

    if args.sensor_id:
        filtered = []
        for row in rows_in:
            sid = row.get("sensor_id")
            if not sid:
                sid, _ = infer_sensor_metadata(row["hdr_path"])
            if sid == args.sensor_id:
                filtered.append(row)
        rows_in = filtered

    for row in rows_in:
        out_dir = scene_out_dir(args.out_dir, MODEL_SLUG, row)
        pred = predict_one(
            row["hdr_path"],
            out_dir,
            data_dir,
            args.t_air,
            args.lambda_min,
            args.lambda_max,
            args.cor_coeff,
            args.idx1,
            args.idx2,
            args.idx3,
            args.idx4,
            not args.no_vis,
            row.get("label_path"),
            args.attenuation_profile,
        )
        rows_out.append({**row, "model": MODEL_SLUG, "prediction_path": str(pred)})

    write_prediction_manifest(Path(args.out_dir) / MODEL_SLUG / "prediction_manifest.csv", rows_out)


if __name__ == "__main__":
    main()
