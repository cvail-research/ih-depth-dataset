# Changelog

**05/20/2026 -- the public release workflow now centers on the overlaid IH-Depth dataset and first-user usability**

- IH-Depth files unpack directly into the raw IH scene folders beside the matching raw `.hdr/.bsq` files.
- Depth labels are released as public `uint16` PNG files.
- `ihd.utils.download_ih` downloads only the raw `.hdr/.bsq` files referenced by the released manifests.
- `ihd.utils.prepare_eval_split` prepares a compact evaluation ground-truth tree from a released split CSV.
- Physics-based baseline reproduction files were removed from the tracked release branch so the public repo stays focused on dataset setup, evaluation, and the released HSI learning baselines.

**04/18/2026 -- the repository was cleaned up for public release**

- Dataset-construction artifacts were removed.
- The public surface was narrowed to instructions, benchmarking, and baselines.
