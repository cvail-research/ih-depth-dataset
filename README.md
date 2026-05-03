# ih-depth-dataset

Standalone repository for curating the DARPA Invisible Headlights LWIR-LiDAR
dataset expansion used for registration, annotation, and release preparation.

Short project name: `IH-Depth`.

Its scope is limited to:

- syncing and materializing relevant DARPA Invisible Headlights assets
- preprocessing HiResLIDAR point clouds for annotation
- running the annotation workspaces
- fitting and validating cylindrical camera registrations
- generating correspondence guides and dataset-quality summaries
- preparing manifests and release-oriented metadata for the curated expansion

## Scope

The target artifact is a curated LWIR-LiDAR dataset expansion where each usable
scene supports projection of LiDAR into LWIR image space through either:

- an existing scene `.cyl` together with verified LiDAR-to-correspondence-frame
  registration, or
- a manually fitted `.cyl` from image/LiDAR correspondences collected in the
  annotation workspace

This repository is for dataset curation only. It is not the benchmark-training
or model-development repository. The intended public-facing reference name is
`IH-Depth`.

## Environment

Use `uv` for dependency management and execution.

```bash
uv sync
```

The code targets Python `3.11+`.

## Layout

- `ihd/annotation_workspace/`: workspace for scenes with an existing `.cyl`
- `ihd/annotation_workspace_nocyl/`: workspace for scenes without a `.cyl`
- `ihd/datasets/`: preprocessing, registration, evaluation, and summary tools
- `ihd/datasets/data_utils/ihdataset/`: manifest and DARPA data utilities
- `scripts/data/`: Slurm jobs for sync/materialization
- `scripts/validation/`: Slurm jobs for preprocessing and annotation workflows
- `configs/annotation/`: example environment files

## Workflow

1. Refresh or build manifests for relevant DARPA scenes.
2. Sync or materialize the required LWHSI and HiResLIDAR assets into `/disk`.
3. Preprocess LAS files for lightweight annotation display clouds.
4. Run the annotation workspace for fresh scenes or verification scenes.
5. Fit or validate registrations and export correspondence artifacts.
6. Summarize scene-level quality and prepare the frozen dataset release.

## Frozen `v0` Manifest

The frozen release manifest should be the single source of truth for which
scenes are included in `v0`, which are deferred, and which are excluded.

Each manifest row is expected to capture:

- scene identity: `collection`, `path`, `step`, `scene_label`
- source modality presence: LWIR, LiDAR, correspondence file, `.cyl`
- curation evidence: number of picked pairs, fitted RMSE, `.cyl` verification RMSE
- release artifacts: projected depth labels, residual CSV, summary JSON, overlays
- QC state: annotator votes, majority verdict, notes, final release decision
- exclusion rationale when a scene is not included

The repository includes a factual scene-manifest builder and a frozen-manifest builder:

```bash
uv run python ihd/datasets/build_scene_facts_manifest_v0.py \
  --with-prior-manifest manifests/archive/legacy_pool_manifests_v0/01_with_prior_cyl_n60.csv \
  --without-prior-manifest manifests/archive/legacy_pool_manifests_v0/02_without_prior_cyl_n246.csv \
  --fitted-manifest manifests/archive/legacy_pool_manifests_v0/03_unified_own_fitted_cyl_n232.csv \
  --output-csv manifests/01_scene_facts_n306.csv \
  --output-summary-json manifests/01_scene_facts_n306_summary.json

uv run python ihd/datasets/build_frozen_manifest_v0.py \
  --scene-facts-manifest manifests/01_scene_facts_n306.csv \
  --quality-manifest manifests/03_scene_quality_manifest_current.csv \
  --cleanup-manifest manifests/04_occlusion_cleanup_manifest_current.csv \
  --output-csv manifests/06_frozen_manifest_v0.csv \
  --output-summary-json manifests/06_frozen_manifest_v0_summary.json
```

The QC review file is one row per reviewer verdict:

- `scene_label`
- `annotator_id`
- `verdict`
- `notes`
- `seconds`

Template: [configs/release/frozen_manifest_qc_reviews_template.csv](/home/guille/ih-depth-dataset/configs/release/frozen_manifest_qc_reviews_template.csv)

The optional overrides file is one row per scene and is useful when the final
paper-facing decision differs from the default rule:

- `scene_label`
- `release_decision`
- `exclusion_reason`
- `annotation_mode`
- `cyl_source`
- `qc_status`
- `qc_notes`

Template: [configs/release/frozen_manifest_overrides_template.csv](/home/guille/ih-depth-dataset/configs/release/frozen_manifest_overrides_template.csv)

### Default Inclusion Rule

By default, the builder marks a scene as:

- `include` if all required artifacts are present and the effective QC verdict is `good`
- `defer` if artifacts are present but the effective QC verdict is `usable with caution` or still missing
- `exclude` if required artifacts are missing or the effective QC verdict is `bad`

The effective QC verdict is taken from the 4-rater majority vote when review
rows are available; otherwise it falls back to the scene `summary.json` verdict.

### Recommended QC Protocol

For the rapid 10-second review you described, keep the rubric intentionally
simple and paper-defensible:

- `good`: labels are release-ready for supervised training/evaluation
- `usable with caution`: structure is mostly right but there are visible issues worth fixing before release
- `bad`: misregistration, depth projection, or scene artifacts make the label unreliable

For `v0`, include only `good` scenes. Keep `usable with caution` scenes in the
manifest as deferred so they remain auditable and can be revisited for a later
version instead of silently disappearing.

## Rapid QC Interface

For fast multi-rater qualitative review, the repository includes a dedicated web
app that discovers every scene with a `fitted_rigid_overlay.png` under
`analysis/lidar_labeling` and merges that with ready scenes from
`analysis/annotation_workspace*`, shows the reference pseudobroadband HSI
beside the overlay, and lets reviewers assign:

- `good`
- `usable with caution`
- `bad`

The interface also provides:

- a per-scene timer that turns red and blinks after 30 seconds
- `Back` and `Next` navigation
- remaining-scene progress
- automatic persistence of verdicts and accumulated viewing time

Reviewer outputs are saved under:

- `analysis/qc_review/sessions/<reviewer_id>/session.json`
- `analysis/qc_review/sessions/<reviewer_id>/reviews.csv`

Reference grayscale previews are cached under:

- `analysis/qc_review/cache/<collection>/<path>/<step>/reference.png`

If you want to make the QC images available directly inside the shared `/disk`
scene folders, stage them with:

```bash
sbatch scripts/validation/run_stage_qc_assets_to_disk.sh
```

This writes dataset-style PNGs into each scene folder such as:

- `IHTest_202104_Path15_Step11_PseudoBB_collect0_DistStA.png`
- `IHTest_202104_Path15_Step11_DepthOverlay_collect0_DistStA.png`

For scenes without explicit collect suffixes, the staged names follow the same
dataset pattern without `collect0`, for example:

- `IHTest_202204_Path20_Step1_PseudoBB_DistStA.png`
- `IHTest_202204_Path20_Step1_DepthOverlay_DistStA.png`

### Platform-Filtered Overlays

To regenerate all preprocessed LiDAR files with the shared platform-removal
sphere, submit one Slurm job:

```bash
sbatch scripts/validation/run_platform_sphere_preprocessing_batch.sh
```

This uses one reproducible sphere for every scene:
`center=(-0.109999, -0.001428, -0.155019), radius=2.5 m`. It writes outputs
under `analysis/lidar_preprocessing/.../*_platform_sphere_r2p5/`.

After that job finishes, refresh the staged depth overlays in the original
`/disk` scene folders:

```bash
sbatch scripts/validation/run_stage_platform_sphere_overlays_to_disk.sh
```

The refreshed overlays keep the same dataset-style filenames, for example
`IHTest_202204_Path33_Step9_DepthOverlay_DistStA.png`.

### Workstation Command

Submit the QC app through Slurm:

```bash
sbatch scripts/validation/run_qc_review_app.sh <reviewer_id>
```

Example:

```bash
sbatch scripts/validation/run_qc_review_app.sh guille
```

The job serves the app on port `8765` by default and logs the assigned node in
`logs/out/<jobid>_ih_qc_review.out`.

### Local Port Forward

From your local machine, forward the QC app port through the workstation login
host to the active Slurm node:

```bash
ssh -N -L 8765:$(ssh YOUR_WORKSTATION_HOST "squeue -n ih_qc_review -h -o %N | head -n 1"):8765 YOUR_WORKSTATION_HOST
```

Then open:

```text
http://localhost:8765
```

If you need a non-default port or a different results root, the Slurm launcher
also accepts:

```bash
sbatch scripts/validation/run_qc_review_app.sh <reviewer_id> <results_root> <port>
```

## Notes

- Heavy jobs should be submitted through the Slurm scripts in `scripts/`.
- This repo assumes the workstation stores dataset assets under `/disk`.
- Results are written under `analysis/` at the repo root and are ignored by git.
