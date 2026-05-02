import json
import threading
from pathlib import Path
from typing import Any

import laspy
import numpy as np

from ihd.annotation_workspace.scene_service import REPO_ROOT, _now, load_gray_preview
from ihd.annotation_workspace_nocyl.scene_service import NoCylSceneWorkspace
from ihd.datasets.cylindrical_camera import project_vect_safe, read_cam
from ihd.datasets.depth_rasterization import depth_range, rasterize
from ihd.datasets.preprocess_las_for_projection import write_subset_las
from ihd.annotation_workspace.scene_service import save_overlay


WORKSPACE_ROOT_CLEANUP = REPO_ROOT / "analysis" / "occlusion_cleanup_workspace"


class OcclusionCleanupWorkspace:
    def __init__(self, collection: str, path_name: str, step: int | str):
        self.source = NoCylSceneWorkspace(collection, path_name, step)
        self.collection = self.source.collection
        self.path_name = self.source.path_name
        self.step = self.source.step
        self.scene_dir = self.source.scene_dir
        self.path_key = self.source.path_key
        self.scene_key = self.source.scene_key
        self.workspace_dir = WORKSPACE_ROOT_CLEANUP / self.collection / self.path_key / self.scene_key
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        self.cleanup_preview_json_path = self.workspace_dir / "cleanup_preview.json"
        self.cleanup_raw_overlay_path = self.workspace_dir / "raw_overlay.png"
        self.cleanup_overlay_path = self.workspace_dir / "cleanup_overlay.png"
        self.cleanup_clean_las_path = self.workspace_dir / f"{self.scene_key}_cleanup_projection_clean.las"

        self._lock = threading.Lock()

    @property
    def hdr_path(self) -> Path:
        return self.source.hdr_path

    @property
    def image_preview_path(self) -> Path:
        return self.source.image_preview_path

    @property
    def projection_las(self) -> Path | None:
        return self.source.projection_las

    @property
    def fit_json_path(self) -> Path:
        return self.source.fit_json_path

    @property
    def fitted_cyl_path(self) -> Path:
        return self.source.fitted_cyl_path

    def prepare(self) -> None:
        self.source.prepare()

    def get_scene_payload(self) -> dict[str, Any]:
        scene = self.source.get_scene_payload()
        scene["cleanup_workspace_dir"] = str(self.workspace_dir)
        scene["cleanup_preview"] = self._load_cleanup_preview()
        scene["cleanup_available"] = bool(self.get_fit_status().get("ready"))
        return scene

    def get_session(self) -> dict[str, Any]:
        return self.source.get_session()

    def get_pointcloud_payload(self) -> dict[str, Any]:
        return self.source.get_pointcloud_payload()

    def get_fit_status(self) -> dict[str, Any]:
        return self.source.get_fit_status()

    def compute_fit(self) -> dict[str, Any]:
        return self.source.compute_fit()

    def _load_cleanup_preview(self) -> dict[str, Any] | None:
        if not self.cleanup_preview_json_path.exists():
            return None
        return json.loads(self.cleanup_preview_json_path.read_text())

    def preview_cleanup(self, center_xyz: list[float], half_extent_m: float = 1.0) -> dict[str, Any]:
        if len(center_xyz) != 3:
            raise ValueError("Cleanup preview center must contain exactly three coordinates.")
        if half_extent_m <= 0:
            raise ValueError("Cleanup preview half extent must be positive.")
        fit_state = self.get_fit_status()
        if not fit_state.get("ready"):
            raise ValueError("Cleanup preview requires an existing fitted overlay.")
        if self.projection_las is None:
            raise FileNotFoundError("Preprocessed LAS is not available for this scene.")

        center = np.asarray(center_xyz, dtype=np.float64)
        las = laspy.read(self.projection_las)
        xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
        if xyz.size == 0:
            raise ValueError("Cleanup preview cannot run on an empty cloud.")

        removed_mask = np.all(np.abs(xyz - center.reshape(1, 3)) <= float(half_extent_m), axis=1)
        keep_idx = np.flatnonzero(~removed_mask)
        if keep_idx.size == 0:
            raise ValueError("Cleanup preview would remove all points; widen the box or choose a different center.")

        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        write_subset_las(las, keep_idx, self.cleanup_clean_las_path)
        self._render_overlay_for_las(
            self.projection_las,
            self.cleanup_raw_overlay_path,
            f"{self.path_name.split('_DistStA')[0]} Step{self.step} | raw overlay",
        )
        self._render_overlay_for_las(
            self.cleanup_clean_las_path,
            self.cleanup_overlay_path,
            f"{self.path_name.split('_DistStA')[0]} Step{self.step} | cleanup overlay",
        )

        preview = {
            "scene_key": self.scene_key,
            "center_xyz": [float(v) for v in center.tolist()],
            "half_extent_m": float(half_extent_m),
            "source_projection_las": str(self.projection_las),
            "cleaned_las": str(self.cleanup_clean_las_path),
            "raw_overlay": str(self.cleanup_raw_overlay_path),
            "cleanup_overlay": str(self.cleanup_overlay_path),
            "removed_points": int(removed_mask.sum()),
            "kept_points": int(keep_idx.size),
            "fit_rmse_total_px": float(fit_state.get("fit_rmse_total", np.nan)),
            "updated_at": _now(),
        }
        self.cleanup_preview_json_path.write_text(json.dumps(preview, indent=2, sort_keys=True) + "\n")
        return preview

    def _render_overlay_for_las(self, las_path: Path, out_path: Path, title: str) -> None:
        gray = load_gray_preview(self.hdr_path).astype(np.float64) / 255.0
        cam = read_cam(str(self.fitted_cyl_path))
        las = laspy.read(las_path)
        xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
        Pc = (cam.Rot @ xyz.T).T + cam.t.reshape(1, 3)
        d = depth_range(Pc)
        ij = project_vect_safe(xyz, cam)
        i_vals = ij[:, 0]
        j_vals = ij[:, 1]
        valid = np.isfinite(i_vals) & np.isfinite(j_vals) & np.isfinite(d)
        H, W = gray.shape
        if np.any(valid):
            i_vals = i_vals[valid]
            j_vals = j_vals[valid]
            d = d[valid]
            inside = (i_vals >= 0) & (i_vals < W) & (j_vals >= 0) & (j_vals < H)
            i_vals = i_vals[inside].astype(np.float32)
            j_vals = j_vals[inside].astype(np.float32)
            d = d[inside].astype(np.float32)
        else:
            i_vals = np.empty((0,), dtype=np.float32)
            j_vals = np.empty((0,), dtype=np.float32)
            d = np.empty((0,), dtype=np.float32)
        depth_img = rasterize(W, H, i_vals, j_vals, d)
        save_overlay(gray, depth_img, out_path, title)
