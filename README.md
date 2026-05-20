# IH-Depth

## Overview

[IH-Depth](https://huggingface.co/datasets/SemilleroCV/ih-depth) is a curated LWIR/LWHSI-LiDAR benchmark derived from the [Invisible Headlights (IH) dataset](https://registry.opendata.aws/darpa-invisible-headlights/). The released benchmark contains 51 off-road scenes, split into 41 training scenes and 10 test scenes. The dataset is released under the CC BY 4.0 license. This repository is the benchmark devkit and code for running the reference baseline methods for that release.

![IH-Depth dataset](assets/ih_depth_dataset.jpg?raw=true)

## Getting Started

### Download the dataset

First download and unpack the raw [Invisible Headlights (IH) dataset](https://registry.opendata.aws/darpa-invisible-headlights/). Then download [IH-Depth](https://huggingface.co/datasets/SemilleroCV/ih-depth) and unpack it into the same root directory.

IH-Depth is a KITTI-style overlay on top of the raw IH tree. The benchmark depth maps, cylindrical camera files, and correspondence files follow the raw IH folder structure, so each IH-Depth scene artifact lands beside the corresponding raw LWHSI `.hdr/.bsq` files:

```text
RAW_IH_ROOT/
  <collection>/
    <PathXX_DistStA>/
      <PathXX_StepYY_DistStA>/
        <raw_lwhsi_stem>.hdr
        <raw_lwhsi_stem>.bsq
        <raw_lwhsi_stem>_depth.png
        <raw_lwhsi_stem>.cyl
        <raw_lwhsi_stem>_corresp.txt
```

The IH-Depth release root also contains `scenes_train.csv`, `scenes_test.csv`, `scenes_manifest.csv`, and `release_summary.json`. The train/test split CSVs define the IH-Depth benchmark splits; they are not part of the raw IH dataset.

Here, `<raw_lwhsi_stem>` means the original LWHSI filename stem for that scene. For example, if the original LWHSI file is named:

```text
IHTest_202104_Path15_Step11_LWHSI1_collect0_DistStA.hdr
```

then depth PNG must be named:

```text
IHTest_202104_Path15_Step11_LWHSI1_collect0_DistStA_depth.png
```

the same applies for other artifacts.

### IH-Depth train and test

For IH-Depth training and testing sets we release sparse LiDAR-projected metric depth labels, per-scene cylindrical camera geometry, and correspondence files. Each released scene has exactly these public benchmark artifacts, all sharing the original raw LWHSI stem:

```text
<raw_lwhsi_stem>_depth.png
<raw_lwhsi_stem>.cyl
<raw_lwhsi_stem>_corresp.txt
```

Depth PNGs follow KITTI-style encoding contract with a small modification:

```text
dtype: uint16
stored_value = round(128 * depth_m)
0 = invalid / unlabeled
depth_m = stored_value / 128
```

This `x128` scale preserves the released maximum observed depth range (<`415 m`) while remaining KITTI-style 16-bit PNG encoding format. `ihd/example/` contains the first-training-scene artifact bundle: a benchmark-format depth PNG plus the matching `.cyl` and correspondence `.txt` example files. The example is for structure and inspection.

## Test Evaluation

We provide an evaluator to compute common depth estimation metrics (AbsRel, RMSE, $\delta$ and more) on the IH-Depth test set. First set up the benchmark environment:

```bash
uv sync
```

After unpacking IH-Depth over the raw IH root, create a compact ground-truth evaluation folder from `scenes_test.csv`:

```bash
uv run python -m ihd.utils.prepare_eval_split RAW_IH_ROOT GT_DIR --split_csv scenes_test.csv
```

Use `--symlink` to link depth PNGs instead of copying them. The helper only prepares depth PNGs because the evaluator does not consume `.cyl` or `_corresp.txt` files.

Run your model on the raw `.hdr/.bsq` input files and save prediction PNGs into a mirrored prediction tree:

```text
GT_DIR/                                           # ground-truth root folder
├── <collection>/
│   ├── <PathXX_DistStA>/
│   │   ├── <PathXX_StepYY_DistStA>/
│   │   │   ├── <raw_lwhsi_stem>_depth.png       # ground-truth depth PNG
│   │   │   └── ...
│   │   └── ...
│   └── ...
└── ...

PREDICTION_DIR/                                   # prediction root folder
├── <collection>/
│   ├── <PathXX_DistStA>/
│   │   ├── <PathXX_StepYY_DistStA>/
│   │   │   ├── <raw_lwhsi_stem>_depth.png       # prediction depth PNG
│   │   │   └── ...
│   │   └── ...
│   └── ...
└── ...
```

Then run the evaluator:

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

If you also want to run the retained `Depth Anything V2` baseline example below, install its extra dependencies with:

```bash
uv sync --extra depthanythingv2
```

Then, make inference with the following command:

```bash
uv run python -m baselines.learning_broadband.depthanythingv2 \
  --hdr /path/to/scene/IHTest_202104_Path15_Step11_LWHSI1_collect0_DistStA.hdr \
  --out-dir /tmp/ihd_predictions \
  --device cuda
```

That command writes a public-contract prediction PNG named `<raw_lwhsi_stem>_depth.png` that can be dropped into `PREDICTION_DIR`.

## Acknowledgement

We sincerely thank the authors of [Concurrent Band Selection and Traversability Estimation From Long-Wave Hyperspectral Imagery in Off-Road Settings](https://openaccess.thecvf.com/content/WACV2024/papers/Yellin_Concurrent_Band_Selection_and_Traversability_Estimation_From_Long-Wave_Hyperspectral_Imagery_WACV_2024_paper.pdf) for releasing the [IH dataset](https://registry.opendata.aws/darpa-invisible-headlights/). We are grateful to the authors of [Ozone-Cues-Mitigate-Reflected-Downwelling-Radiance-in-LWIR-Absorption-Based-Ranging](https://github.com/unaydorken/Ozone-Cues-Mitigate-Reflected-Downwelling-Radiance-in-LWIR-Absorption-Based-Ranging), [Depth Pro](https://github.com/apple/ml-depth-pro), [UniK3D](https://github.com/lpiccinelli-eth/UniK3D), [UniDepth](https://github.com/lpiccinelli-eth/unidepth), [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), and [KITTI-devkit](https://github.com/joseph-zhong/KITTI-devkit) for open-sourcing their code and models.

## License

The code in this repository is released under the MIT license in `LICENSE`. The IH-Depth benchmark data released [here](https://huggingface.co/datasets/SemilleroCV/ih-depth) is released under CC BY 4.0.
