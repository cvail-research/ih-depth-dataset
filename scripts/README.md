# Scripts

This repository keeps only the scripts needed for DARPA Invisible Headlights
dataset curation and annotation.

## Layout

- `data/`: sync, manifest refresh, and materialization jobs
- `evaluation/`: depth prediction, metric evaluation, and split-construction jobs
- `validation/`: LAS preprocessing, annotation workspace, guide generation, and
  registration-processing jobs

## Notes

- Submit Slurm jobs from the repo root.
- Outputs are written to `logs/` and `analysis/`.
- The scripts assume the workstation dataset root is `/disk`.
- Do not run model inference on login nodes. Use the Slurm wrappers in
  `scripts/evaluation/`.

## Baseline Prediction Smoke Test

Build a tiny HDR/label input manifest:

```bash
scripts/evaluation/run_build_prediction_input_manifest.sh \
  --scene-manifest analysis/qc_review/reproducible_qc_report/scenes_accepted_by_rmse5px_distance_5pct_with_drop_rule.csv \
  --limit 3 \
  --out-csv analysis/evaluation/baseline_smoke_predictions/prediction_inputs.csv
```

Check which model environments are available:

```bash
scripts/evaluation/run_check_baseline_model_envs.sh
```

Run one model through Slurm:

```bash
sbatch scripts/evaluation/run_one_baseline_model_predictions.sh \
  unik3d \
  analysis/qc_review/reproducible_qc_report/scenes_accepted_by_rmse5px_distance_5pct_with_drop_rule.csv \
  3 \
  analysis/evaluation/baseline_smoke_predictions
```

Each model writes `depth_prediction.npz` files with a `depth_m` key plus
`metrics_per_scene.csv`, which can be merged into the hard-split score table:

```bash
scripts/evaluation/run_merge_baseline_metrics.sh \
  --predictions-root analysis/evaluation/baseline_smoke_predictions \
  --out-csv analysis/evaluation/baseline_scene_scores.csv
```
