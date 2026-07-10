# Changelog

**07/10/2026 -- geometry and correspondence artifacts are restored for the public release**

- IH-Depth `.cyl` cylindrical camera geometry and `_corresp.txt` correspondence files are restored as released benchmark artifacts for each public scene.
- The released scene manifests include `cyl_relpath` and `corresp_relpath` columns beside `depth_png_relpath`.
- We verified that the restored `.cyl` and `_corresp.txt` files do not overwrite raw IH geometry/correspondence files for the released scenes.

**05/22/2026 -- the public release is now depth-label-only**

- Public IH-Depth release artifacts are now limited to benchmark `*_depth.png` files plus split manifests.
- Public `.cyl` and `_corresp.txt` files were removed from the active release contract.
- `release_summary.json` was removed from the public release surface.
- Public .cyl/_corresp.txt are deferred until there is a coherent release story that does not mutate or partially replace raw IH geometry/correspondence files.

**05/20/2026 -- the public release workflow now centers on the overlaid IH-Depth dataset and first-user usability**

- IH-Depth files unpack directly into the raw IH scene folders beside the matching raw `.hdr/.bsq` files.
- Depth labels are released as public `uint16` PNG files.
- `ihd.utils.download_ih` downloads only the raw `.hdr/.bsq` files referenced by the released manifests.
- `ihd.utils.prepare_eval_split` prepares a compact evaluation ground-truth tree from a released split CSV.
- Physics-based baseline reproduction files were removed from the tracked release branch so the public repo stays focused on dataset setup, evaluation, and the released HSI learning baselines.

**04/18/2026 -- the repository was cleaned up for public release**

- Dataset-construction artifacts were removed.
- The public surface was narrowed to instructions, benchmarking, and baselines.
