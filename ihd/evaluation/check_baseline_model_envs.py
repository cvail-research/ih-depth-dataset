from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


MODELS = {
    "unik3d": ["torch", "spectral", "pkg_resources", "unik3d"],
    "unidepthv2": ["torch", "spectral", "unidepth"],
    "depthanythingv2": ["torch", "spectral", "transformers"],
    "depthpro": ["torch", "spectral", "depth_pro"],
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Check Python environments for IH baseline model runners.")
    ap.add_argument("--out-csv", default="analysis/evaluation/baseline_model_env_check.csv")
    ap.add_argument("--unik3d-python", default="/home/guille/.conda/envs/deeptr/bin/python")
    ap.add_argument("--unidepthv2-python", default="python")
    ap.add_argument("--depthanythingv2-python", default="python")
    ap.add_argument("--depthpro-python", default="python")
    return ap.parse_args()


def check_module(python_bin: str, module: str) -> tuple[bool, str]:
    code = f"import {module}; print('ok')"
    proc = subprocess.run(
        [python_bin, "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    ok = proc.returncode == 0
    message = (proc.stdout if ok else proc.stderr).strip().splitlines()
    return ok, message[-1] if message else ""


def main() -> None:
    args = parse_args()
    python_by_model = {
        "unik3d": args.unik3d_python,
        "unidepthv2": args.unidepthv2_python,
        "depthanythingv2": args.depthanythingv2_python,
        "depthpro": args.depthpro_python,
    }
    rows = []
    for model, modules in MODELS.items():
        python_bin = python_by_model[model]
        for module in modules:
            ok, message = check_module(python_bin, module)
            rows.append(
                {
                    "model": model,
                    "python_bin": python_bin,
                    "module": module,
                    "available": ok,
                    "message": message,
                }
            )
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "python_bin", "module", "available", "message"])
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        status = "OK" if row["available"] else "MISSING"
        print(f"{row['model']:16s} {row['module']:14s} {status} ({row['python_bin']})")


if __name__ == "__main__":
    main()
