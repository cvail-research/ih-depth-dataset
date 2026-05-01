# Training

This folder contains human-facing notes for IH-Depth training runs. Importable
training code lives under `ihd/training`, and Slurm/local launchers live under
`scripts/train`.

## Baseline Depth Anything V2

The first training target is Depth Anything V2 fine-tuned from pseudo-broadband
LWHSI inputs to projected LiDAR depth labels.

Expected input manifests are CSV files with at least:

- `hdr_path`: ENVI LWHSI `.hdr` file.
- `label_path`: projected LiDAR depth label `.npz` with `depth_m` and `valid_mask`.

Launch on Slurm from the repo root:

```bash
sbatch scripts/train/baseline/submit_train_depthanythingv2.sh \
  path/to/train_manifest.csv \
  path/to/val_manifest.csv \
  analysis/training/depthanythingv2/first_run
```

The training script uses the same pseudo-broadband conversion as the evaluation
baseline runners: sum all hyperspectral bands, min-max normalize, and replicate
to RGB.

Outputs include:

- `config.json`
- `final_metrics.json`
- `checkpoints/step_*/`
- `previews/step_*/prediction.png`
- `previews/step_*/target.png`

Weights & Biases logging is optional. The Slurm wrapper defaults to
`WANDB_ENTITY=ai-uis` and `WANDB_PROJECT=ih-depth`; set `WANDB_MODE=disabled`
or pass `--wandb-mode disabled` after the output directory to disable logging.

## UniK3D

UniK3D uses the same manifest format and pseudo-broadband input encoding. Its
training script mirrors UniK3D inference preprocessing, calls the model decoder,
and optimizes the predicted metric `depth` with SiLog loss.

Launch on Slurm from the repo root:

```bash
sbatch scripts/train/baseline/submit_train_unik3d.sh \
  path/to/train_manifest.csv \
  path/to/val_manifest.csv \
  analysis/training/unik3d/first_run
```

## UniDepthV2

UniDepthV2 uses the same manifest format and pseudo-broadband input encoding.
Its trainer mirrors the UniDepthV2 inference preprocessing and optimizes the
predicted metric depth with SiLog loss.

Launch on Slurm from the repo root:

```bash
sbatch scripts/train/baseline/submit_train_unidepthv2.sh \
  path/to/train_manifest.csv \
  path/to/val_manifest.csv \
  analysis/training/unidepthv2/first_run
```

## HSI Depth Anything V2

The HSI trainer keeps the full LWHSI cube and adapts the Depth Anything V2 patch
embedding from 3 input channels to `B` hyperspectral bands. It uses the same
manifest format, projected depth labels, SiLog loss, checkpoints, previews, and
optional W&B logging as the baseline trainer.

Launch on Slurm from the repo root:

```bash
sbatch scripts/train/hsi/submit_train_depthanythingv2_hsi.sh \
  path/to/train_manifest.csv \
  path/to/val_manifest.csv \
  analysis/training/depthanythingv2_hsi/first_run
```

For one-scene overfit tests or very small manifests, preload normalized HSI
tensors in memory:

```bash
sbatch scripts/train/hsi/submit_train_depthanythingv2_hsi.sh \
  path/to/train_manifest.csv \
  path/to/val_manifest.csv \
  analysis/training/depthanythingv2_hsi/overfit_one_scene \
  --cache-mode memory --preload-cache --num-workers 0
```

For repeated larger experiments, cache normalized HSI tensors on disk:

```bash
sbatch scripts/train/hsi/submit_train_depthanythingv2_hsi.sh \
  path/to/train_manifest.csv \
  path/to/val_manifest.csv \
  analysis/training/depthanythingv2_hsi/full_run \
  --cache-mode disk --cache-dir analysis/training/cache/hsi_tensors
```
