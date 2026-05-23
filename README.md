# IH-Depth

## Overview

[IH-Depth](https://huggingface.co/datasets/SemilleroCV/ih-depth) is a curated LWIR/LWHSI-LiDAR benchmark derived from the [Invisible Headlights (IH) dataset](https://registry.opendata.aws/darpa-invisible-headlights/). The released benchmark contains 51 off-road scenes, split into 41 training scenes and 10 test scenes. The dataset is released under the CC BY 4.0 license. This repository provides the benchmark devkit and evaluator for that release.

![IH-Depth dataset](assets/ih_depth_dataset.jpg?raw=true)

## Getting Started

### Download the dataset

Start by preparing a raw IH root directory, referred to below as `RAW_IH_ROOT`.

If you only want the raw `.hdr/.bsq` files needed by the released scenes, download those raw files first with:

```bash
uv run python -m ihd.utils.download_ih RAW_IH_ROOT --manifest metadata/scenes_manifest.csv
```

If you prefer the full raw IH dataset instead of the manifest-limited subset, download and place it into the same `RAW_IH_ROOT` directory.

Then download [IH-Depth](https://huggingface.co/datasets/SemilleroCV/ih-depth).

The public benchmark artifact is a depth PNG stored beside the corresponding raw LWHSI `.hdr/.bsq` files:

```text
RAW_IH_ROOT/
  <collection>/
    <PathXX_DistStA>/
      <PathXX_StepYY_DistStA>/
        <raw_lwhsi_stem>.hdr
        <raw_lwhsi_stem>.bsq
        <raw_lwhsi_stem>_depth.png
```

Here, `<raw_lwhsi_stem>` means the original LWHSI filename stem for that scene. For example, if the original LWHSI file is named:

```text
IHTest_202104_Path15_Step11_LWHSI1_collect0_DistStA.hdr
```

then the released depth PNG must be named:

```text
IHTest_202104_Path15_Step11_LWHSI1_collect0_DistStA_depth.png
```

If you downloaded IH-Depth into a separate directory `IH_DEPTH_ROOT`, place only the depth PNGs into `RAW_IH_ROOT` with:

```bash
rsync -a --include '*/' --include '*_depth.png' --exclude '*' IH_DEPTH_ROOT/ RAW_IH_ROOT/
```

The IH-Depth release root contains `scenes_train.csv`, `scenes_test.csv`, and `scenes_manifest.csv`. The train/test split CSVs define the IH-Depth benchmark splits; they are not part of the raw IH dataset.

### IH-Depth train and test

For IH-Depth training and testing sets we release sparse LiDAR-projected metric depth labels. Each released scene has the following public benchmark artifact:

```text
<raw_lwhsi_stem>_depth.png
```

Depth PNGs follow KITTI-style encoding contract with a small modification:

```text
dtype: uint16
stored_value = round(128 * depth_m)
0 = invalid / unlabeled
depth_m = stored_value / 128
```

This `x128` scale preserves the released maximum observed depth range (<`415 m`) while remaining 16-bit PNG encoding format. `ihd/example/` contains a benchmark-format depth PNG example for structure and inspection.

## Test Evaluation

We provide an evaluator to compute common depth estimation metrics (AbsRel, RMSE, $\delta$ and more) on the IH-Depth test set. First set up the benchmark environment:

```bash
uv sync
```

After placing the benchmark depth PNGs into the raw IH root, create a compact ground-truth evaluation folder from `scenes_test.csv`:

```bash
uv run python -m ihd.utils.prepare_eval_split RAW_IH_ROOT GT_DIR --split_csv scenes_test.csv
```

Use `--symlink` to link depth PNGs instead of copying them.

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

If you also want to run hyperspectral baselines, go [here](baselines/).

## Acknowledgement

We sincerely thank the authors of [Concurrent Band Selection and Traversability Estimation From Long-Wave Hyperspectral Imagery in Off-Road Settings](https://openaccess.thecvf.com/content/WACV2024/papers/Yellin_Concurrent_Band_Selection_and_Traversability_Estimation_From_Long-Wave_Hyperspectral_Imagery_WACV_2024_paper.pdf) for releasing the [IH dataset](https://registry.opendata.aws/darpa-invisible-headlights/). We are grateful to the authors of [KITTI-devkit](https://github.com/joseph-zhong/KITTI-devkit) for open-sourcing their evaluation code.

## License

The code in this repository is released under the MIT license in `LICENSE`. The IH-Depth benchmark data released [here](https://huggingface.co/datasets/SemilleroCV/ih-depth) is released under CC BY 4.0.
