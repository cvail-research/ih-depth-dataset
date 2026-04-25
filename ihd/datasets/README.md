# IHD Datasets

This package contains the dataset curation utilities for `IH-Depth`.

Its focus is the DARPA Invisible Headlights LWIR-LiDAR expansion workflow:

- building and refreshing manifests for relevant DARPA scenes
- syncing LWHSI and HiResLIDAR assets from the public dataset
- preprocessing LAS files for annotation-friendly point clouds
- fitting or validating cylindrical camera registrations
- exporting correspondence artifacts, summaries, and quality-control outputs

The code in this directory is intentionally limited to dataset curation and
release preparation. It does not include model-training or unrelated legacy
dataset pipelines.
