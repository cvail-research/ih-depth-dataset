# IH-Depth

## Overview

This repository is the IH-Depth benchmark devkit and baseline runner repository.

## Getting Started

Download the released dataset assets from the Hugging Face dataset repository `SemilleroCV/ih-depth`, then arrange benchmark evaluation files with mirrored ground-truth and prediction trees:

```text
GT_DIR/
  <collection>/<PathXX_DistStA>/<PathXX_StepYY_DistStA>/
    <raw_lwhsi_stem>_depth.png

PREDICTION_DIR/
  <collection>/<PathXX_DistStA>/<PathXX_StepYY_DistStA>/
    <raw_lwhsi_stem>_depth.png
```

Example evaluated basename:

```text
IHTest_202104_Path15_Step11_LWHSI1_collect0_DistStA_depth.png
```

Depth PNGs use the public IH-Depth encoding contract:

```text
dtype: uint16
stored_value = round(128 * depth_m)
0 = invalid / unlabeled
depth_m = stored_value / 128
```

This `x128` scale preserves the released maximum observed depth range (`415.993591 m`) while remaining KITTI-style in format and semantics. `ihd/example/` contains one real first-training-scene artifact bundle: a benchmark-format depth PNG plus the matching `.cyl` and correspondence `.txt` example files. The example is for structure and inspection; evaluator inputs still follow the mirrored `<raw_lwhsi_stem>_depth.png` benchmark contract above.

You can inspect the example depth PNG directly from Python:

```bash
uv run python -c "from ihd.utils.depth_png import load_depth_png; d, m = load_depth_png('ihd/example/IHTest_202009_Path1_Step2_LWHSI1__DistStA_depth.png'); print(d.shape, int(m.sum()), float(d[m].max()) if m.any() else 0.0)"
```

## Test Evaluation

Run benchmark evaluation with the public entrypoint:

```bash
uv run python ihd/ihd_evaluator.py GT_DIR PREDICTION_DIR --output_name stats_ihd.txt
```

The evaluator writes `stats_ihd.txt` inside `PREDICTION_DIR`, prints the aggregate summary to stdout, and emits:

```text
PREDICTION_DIR/
  stats_ihd.txt
  errors_out/
  errors_img/
  depth_gt/
  depth_pred/
  input_preview/
```

`input_preview/` is best-effort and is skipped per scene when an optional preview source is unavailable. One retained baseline inference example is:

```bash
uv run python -m baselines.learning_broadband.depthanythingv2 \
  --hdr /path/to/scene/IHTest_202104_Path15_Step11_LWHSI1_collect0_DistStA.hdr \
  --out-dir /tmp/ihd_predictions \
  --device cuda
```

That command writes a public-contract prediction PNG named `<raw_lwhsi_stem>_depth.png` that can be dropped into `PREDICTION_DIR`.

## Acknowledgement

We sincerely thank the authors of [Ozone-Cues-Mitigate-Reflected-Downwelling-Radiance-in-LWIR-Absorption-Based-Ranging](https://github.com/unaydorken/Ozone-Cues-Mitigate-Reflected-Downwelling-Radiance-in-LWIR-Absorption-Based-Ranging), [Depth Pro](https://github.com/apple/ml-depth-pro), [UniK3D](https://github.com/lpiccinelli-eth/UniK3D), [UniDepth](https://github.com/lpiccinelli-eth/unidepth), [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), and [KITTI-devkit](https://github.com/joseph-zhong/KITTI-devkit) for open-sourcing their code and models.

## License

Repository code is released under the MIT license in `LICENSE`. The IH-Depth dataset license and source distribution terms are separate from this repository and remain governed by the dataset release.
