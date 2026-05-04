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

## Visible Scene Categorization Prep (Split v0)

Build visible-asset resolution for split scenes (exact BBVIS first, nearest-step fallback in same path, preferred side = LEFT):

```bash
scripts/validation/run_build_visible_scene_manifest.sh \
  --split-manifest manifests/07_split_definition_v0/scene_splits.csv \
  --output-csv manifests/07_split_definition_v0/scene_splits_visible_resolution_v0.csv \
  --missing-csv manifests/07_split_definition_v0/scene_splits_visible_missing_v0.csv \
  --summary-json manifests/07_split_definition_v0/scene_splits_visible_resolution_summary_v0.json
```

Download missing BBVIS assets in parallel chunks (via Slurm):

```bash
sbatch scripts/data/run_sync_ih_scene_bbvis_chunk.sh manifests/07_split_definition_v0/bbvis_missing_chunks/chunk_01.csv
```

After jobs finish, rerun the visible-resolution command to refresh unresolved scenes.
