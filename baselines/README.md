# Hyperspectral Baselines

These scripts run learning-based monocular depth models directly on raw IH LWHSI `.hdr/.bsq` inputs by replacing each model's RGB patch projection with a hyperspectral projection.

Available scripts:

```text
baselines/depthanythingv2_hsi.py
baselines/unidepthv2_hsi.py
baselines/unik3d_hsi.py
```

Install the base devkit first:

```bash
uv sync
```

Then install the extra dependency for the baseline you want to run:

```bash
uv sync --extra depthanythingv2
uv sync --extra unidepthv2
uv sync --extra unik3d
```

Run one scene:

```bash
uv run python -m baselines.unidepthv2_hsi \
  --hdr RAW_IH_ROOT/<collection>/<PathXX_DistStA>/<PathXX_StepYY_DistStA>/<raw_lwhsi_stem>.hdr \
  --out-dir PREDICTION_DIR \
  --device cuda
```

The script writes a prediction PNG using the public IH-Depth encoding:

```text
stored_value = round(128 * depth_m)
0 = invalid
```

To evaluate, place or copy the generated `<raw_lwhsi_stem>_depth.png` into the same mirrored structure used by `GT_DIR`, then run:

```bash
uv run python ihd/ihd_evaluator.py GT_DIR PREDICTION_DIR
```

## Acknowledgement

These baseline scripts adapt open-source monocular depth models to IH LWHSI inputs. We thank the authors of [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), [UniDepth](https://github.com/lpiccinelli-eth/unidepth), and [UniK3D](https://github.com/lpiccinelli-eth/UniK3D) for releasing their code and model weights.
