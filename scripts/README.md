# Scripts

This repository keeps only the scripts needed for DARPA Invisible Headlights
dataset curation and annotation.

## Layout

- `data/`: sync, manifest refresh, and materialization jobs
- `validation/`: LAS preprocessing, annotation workspace, guide generation, and
  registration-processing jobs

## Notes

- Submit Slurm jobs from the repo root.
- Outputs are written to `logs/` and `analysis/`.
- The scripts assume the workstation dataset root is `/disk`.
