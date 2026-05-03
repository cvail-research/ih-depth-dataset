import csv
import json
import threading
import uuid
from pathlib import Path
from typing import Any

import laspy
import numpy as np

from ihd.annotation_workspace.scene_service import BROWSER_POINT_BUDGET, REPO_ROOT, _now, load_gray_preview
from ihd.annotation_workspace_nocyl.scene_service import NoCylSceneWorkspace
from ihd.datasets.cylindrical_camera import camera, project_vect_safe, read_cam
from ihd.datasets.depth_rasterization import depth_range, rasterize
from ihd.datasets.preprocess_las_for_projection import write_subset_las
from ihd.annotation_workspace.scene_service import save_overlay


WORKSPACE_ROOT_CLEANUP = REPO_ROOT / "analysis" / "occlusion_cleanup_workspace"
OCCLUSION_CLEANUP_MANIFEST = REPO_ROOT / "manifests" / "06_occlusion_cleanup_manifest_current.csv"
OCCLUSION_CLEANUP_SUMMARY = REPO_ROOT / "manifests" / "06_occlusion_cleanup_manifest_current_summary.json"


def _normalize_path_name(path_name: str) -> str:
    if path_name.startswith("Path") and path_name.endswith("_DistStA"):
        return path_name
    if path_name.startswith("path") and path_name[4:].isdigit():
        return f"Path{path_name[4:]}_DistStA"
    return path_name


class OcclusionCleanupWorkspace:
    def __init__(self, collection: str, path_name: str, step: int | str):
        normalized_path_name = _normalize_path_name(path_name)
        self.source = NoCylSceneWorkspace(collection, normalized_path_name, step)
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
        self._pointcloud_payload: dict[str, Any] | None = None
        self._pointcloud_cache_key: str | None = None

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
        if self.cleanup_preview_json_path.exists():
            self.sync_cleanup_manifest()

    def get_scene_payload(self) -> dict[str, Any]:
        scene = self.source.get_scene_payload()
        scene["cleanup_workspace_dir"] = str(self.workspace_dir)
        scene["cleanup_preview"] = self._load_cleanup_preview()
        scene["cleanup_available"] = bool(self.get_fit_status().get("ready"))
        return scene

    def get_session(self) -> dict[str, Any]:
        return self.source.get_session()

    def get_pointcloud_payload(self) -> dict[str, Any]:
        source_payload = self.source.get_pointcloud_payload()
        source_las = source_payload.get("source_las")
        display_las_path = Path(source_las) if source_las else None
        display_source = "projection_las"
        preview = self._load_cleanup_preview()
        if preview and self.cleanup_clean_las_path.exists():
            display_las_path = self.cleanup_clean_las_path
            display_source = "cleanup_las"
        if display_las_path is None or not display_las_path.exists():
            raise FileNotFoundError("Display point cloud LAS is not available for cleanup workspace.")

        cache_key = f"{display_las_path}:{display_las_path.stat().st_mtime_ns}"
        if self._pointcloud_payload is not None and self._pointcloud_cache_key != cache_key:
            self._pointcloud_payload = None
        if self._pointcloud_payload is None:
            las = laspy.read(display_las_path)
            xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float32)
            if hasattr(las, "intensity"):
                intensity = np.asarray(las.intensity).astype(np.float32)
                if intensity.size == xyz.shape[0]:
                    finite = np.isfinite(intensity)
                    if np.any(finite):
                        lo = float(np.min(intensity[finite]))
                        hi = float(np.max(intensity[finite]))
                        intensity_norm = (intensity - lo) / (hi - lo) if hi > lo else np.zeros_like(intensity)
                    else:
                        intensity_norm = np.zeros_like(intensity)
                else:
                    intensity_norm = np.zeros((xyz.shape[0],), dtype=np.float32)
            else:
                intensity_norm = np.zeros((xyz.shape[0],), dtype=np.float32)
            original_point_count = int(xyz.shape[0])
            if xyz.shape[0] > BROWSER_POINT_BUDGET:
                stride = int(np.ceil(xyz.shape[0] / BROWSER_POINT_BUDGET))
                keep = np.arange(0, xyz.shape[0], stride, dtype=np.int64)
                xyz = xyz[keep]
                intensity_norm = intensity_norm[keep]

            payload = {
                "source_kind": "display",
                "display_source": display_source,
                "source_las": str(display_las_path),
                "point_count": int(xyz.shape[0]),
                "original_point_count": original_point_count,
                "browser_point_budget": BROWSER_POINT_BUDGET,
                "x": xyz[:, 0].tolist(),
                "y": xyz[:, 1].tolist(),
                "z": xyz[:, 2].tolist(),
                "intensity_norm": intensity_norm.tolist(),
            }
            projected_payload = dict(payload)
            fit_state = self.get_fit_status()
            if fit_state.get("ready"):
                cam_info = fit_state.get("optimized_camera") or fit_state.get("default_camera") or {}
                try:
                    cam = camera(
                        float(cam_info["R"]),
                        float(cam_info["w"]),
                        float(cam_info["y"]),
                        float(cam_info["f"]),
                        float(cam_info["j0"]),
                        np.asarray(cam_info["Rot"], dtype=np.float64),
                        np.asarray(cam_info["t"], dtype=np.float64),
                    )
                    xyz = np.column_stack((payload["x"], payload["y"], payload["z"])).astype(np.float64)
                    projected = project_vect_safe(xyz, cam)
                    depth = depth_range((cam.Rot @ xyz.T).T + cam.t.reshape(1, 3))
                    valid = np.isfinite(projected[:, 0]) & np.isfinite(projected[:, 1]) & np.isfinite(depth)
                    projected_payload["projected_u"] = projected[:, 0].tolist()
                    projected_payload["projected_v"] = projected[:, 1].tolist()
                    projected_payload["projected_depth"] = depth.tolist()
                    projected_payload["projected_valid"] = valid.tolist()
                    projected_payload["projection_image_width"] = int(cam_info.get("image_width", 0) or 0)
                    projected_payload["projection_image_height"] = int(cam_info.get("image_height", 0) or 0)
                except Exception:
                    projected_payload["projected_u"] = []
                    projected_payload["projected_v"] = []
                    projected_payload["projected_depth"] = []
                    projected_payload["projected_valid"] = []
            else:
                projected_payload["projected_u"] = []
                projected_payload["projected_v"] = []
                projected_payload["projected_depth"] = []
                projected_payload["projected_valid"] = []
            self._pointcloud_payload = projected_payload
            self._pointcloud_cache_key = cache_key
        return self._pointcloud_payload

    def get_fit_status(self) -> dict[str, Any]:
        return self.source.get_fit_status()

    def compute_fit(self) -> dict[str, Any]:
        return self.source.compute_fit()

    def _load_cleanup_preview(self) -> dict[str, Any] | None:
        if not self.cleanup_preview_json_path.exists():
            return None
        preview = json.loads(self.cleanup_preview_json_path.read_text())
        return self._normalize_cleanup_preview(preview)

    @staticmethod
    def _normalize_cleanup_preview(preview: dict[str, Any]) -> dict[str, Any]:
        if "regions" in preview:
            regions = []
            for idx, region in enumerate(preview.get("regions") or []):
                normalized_region = dict(region)
                normalized_region.setdefault("region_id", normalized_region.get("region_id") or f"legacy_{idx+1}")
                regions.append(normalized_region)
            normalized = dict(preview)
            normalized["regions"] = regions
            normalized["cleanup_region_count"] = int(preview.get("cleanup_region_count", len(regions)))
            normalized["cleanup_region_ids"] = [str(region.get("region_id")) for region in regions]
            return normalized

        if "center_xyz" not in preview:
            normalized = dict(preview)
            normalized["regions"] = []
            normalized["cleanup_region_count"] = 0
            return normalized

        region = {
            "region_id": f"legacy_1",
            "center_xyz": [float(v) for v in preview.get("center_xyz", [0.0, 0.0, 0.0])],
            "half_extent_m": float(preview.get("half_extent_m", 1.0)),
            "selection_mode": preview.get("selection_mode", "unknown"),
            "updated_at": preview.get("updated_at", _now()),
        }
        normalized = dict(preview)
        normalized["regions"] = [region]
        normalized["cleanup_region_count"] = 1
        normalized["cleanup_region_ids"] = [region["region_id"]]
        normalized["selection_mode_summary"] = {region["selection_mode"]: 1}
        return normalized

    def _cleanup_manifest_row(self, preview: dict[str, Any]) -> dict[str, Any]:
        regions = preview.get("regions") or []
        selection_mode_summary = preview.get("selection_mode_summary")
        if selection_mode_summary is None:
            selection_mode_summary = {}
            for region in regions:
                mode = str(region.get("selection_mode", "unknown"))
                selection_mode_summary[mode] = int(selection_mode_summary.get(mode, 0)) + 1
        last_region = regions[-1] if regions else {}
        return {
            "collection": self.collection,
            "path_key": self.path_key,
            "path_name": self.path_name,
            "step": int(self.step),
            "scene_key": self.scene_key,
            "selection_mode": last_region.get("selection_mode", preview.get("selection_mode", "unknown")),
            "cleanup_status": "previewed",
            "cleanup_region_count": int(preview.get("cleanup_region_count", len(regions))),
            "cleanup_region_ids_json": json.dumps([str(region.get("region_id", "")) for region in regions], sort_keys=True),
            "cleanup_regions_json": json.dumps(regions, sort_keys=True),
            "selection_mode_summary_json": json.dumps(selection_mode_summary, sort_keys=True),
            "center_x_m": float(last_region.get("center_xyz", preview.get("center_xyz", [0.0, 0.0, 0.0]))[0]),
            "center_y_m": float(last_region.get("center_xyz", preview.get("center_xyz", [0.0, 0.0, 0.0]))[1]),
            "center_z_m": float(last_region.get("center_xyz", preview.get("center_xyz", [0.0, 0.0, 0.0]))[2]),
            "half_extent_m": float(last_region.get("half_extent_m", preview.get("half_extent_m", 1.0))),
            "removed_points": int(preview["removed_points"]),
            "kept_points": int(preview["kept_points"]),
            "fit_rmse_total_px": float(preview["fit_rmse_total_px"]),
            "source_projection_las": preview["source_projection_las"],
            "cleaned_las": preview["cleaned_las"],
            "raw_overlay": preview["raw_overlay"],
            "cleanup_overlay": preview["cleanup_overlay"],
            "updated_at": preview["updated_at"],
        }

    @staticmethod
    def _cleanup_manifest_fieldnames() -> list[str]:
        return [
            "collection",
            "path_key",
            "path_name",
            "step",
            "scene_key",
            "selection_mode",
            "cleanup_status",
            "cleanup_region_count",
            "cleanup_region_ids_json",
            "cleanup_regions_json",
            "selection_mode_summary_json",
            "center_x_m",
            "center_y_m",
            "center_z_m",
            "half_extent_m",
            "removed_points",
            "kept_points",
            "fit_rmse_total_px",
            "source_projection_las",
            "cleaned_las",
            "raw_overlay",
            "cleanup_overlay",
            "updated_at",
        ]

    def _rewrite_cleanup_manifest(self, rows: list[dict[str, Any]]) -> None:
        OCCLUSION_CLEANUP_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        rows.sort(key=lambda r: (r["collection"], r["path_key"], int(r["step"])))
        fieldnames = self._cleanup_manifest_fieldnames()
        with OCCLUSION_CLEANUP_MANIFEST.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "manifest_path": str(OCCLUSION_CLEANUP_MANIFEST),
            "scene_count": len(rows),
            "selection_mode_counts": {
                mode: sum(1 for r in rows if r.get("selection_mode") == mode)
                for mode in sorted({r.get("selection_mode") for r in rows})
            },
            "cleanup_status_counts": {
                status: sum(1 for r in rows if r.get("cleanup_status") == status)
                for status in sorted({r.get("cleanup_status") for r in rows})
            },
            "total_removed_points": int(sum(int(r["removed_points"]) for r in rows)) if rows else 0,
            "total_kept_points": int(sum(int(r["kept_points"]) for r in rows)) if rows else 0,
            "mean_removed_points": float(np.mean([int(r["removed_points"]) for r in rows])) if rows else 0.0,
            "mean_kept_points": float(np.mean([int(r["kept_points"]) for r in rows])) if rows else 0.0,
            "updated_at": _now(),
        }
        OCCLUSION_CLEANUP_SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    def sync_cleanup_manifest(self) -> None:
        preview = self._load_cleanup_preview()
        if preview is None:
            return
        row = self._cleanup_manifest_row(preview)
        rows = []
        if OCCLUSION_CLEANUP_MANIFEST.exists():
            with OCCLUSION_CLEANUP_MANIFEST.open("r", newline="") as f:
                reader = csv.DictReader(f)
                rows = [dict(r) for r in reader if dict(r).get("scene_key") != self.scene_key]
        rows.append(row)
        self._rewrite_cleanup_manifest(rows)

    def preview_cleanup(
        self,
        center_xyz: list[float],
        half_extent_m: float = 1.0,
        selection_mode: str = "unknown",
    ) -> dict[str, Any]:
        return self.add_cleanup_region(center_xyz, half_extent_m, selection_mode)

    def recompute_cleanup_preview(self) -> dict[str, Any]:
        preview = self._load_cleanup_preview()
        if preview is None:
            raise ValueError("No cleanup regions exist yet. Add a region first.")
        return self._write_cleanup_preview(self._apply_cleanup_regions(preview.get("regions") or []))

    def _current_regions(self) -> list[dict[str, Any]]:
        preview = self._load_cleanup_preview()
        if preview is None:
            return []
        return list(preview.get("regions") or [])

    def _write_cleanup_preview(self, preview: dict[str, Any]) -> dict[str, Any]:
        self.cleanup_preview_json_path.write_text(json.dumps(preview, indent=2, sort_keys=True) + "\n")
        self.sync_cleanup_manifest()
        return preview

    def _apply_cleanup_regions(self, regions: list[dict[str, Any]]) -> dict[str, Any]:
        fit_state = self.get_fit_status()
        if not fit_state.get("ready"):
            raise ValueError("Cleanup preview requires an existing fitted overlay.")
        if self.projection_las is None:
            raise FileNotFoundError("Preprocessed LAS is not available for this scene.")

        las = laspy.read(self.projection_las)
        xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
        if xyz.size == 0:
            raise ValueError("Cleanup preview cannot run on an empty cloud.")

        if not regions:
            raise ValueError("Cleanup preview requires at least one cleanup region.")

        normalized_regions: list[dict[str, Any]] = []
        removed_mask = np.zeros(xyz.shape[0], dtype=bool)
        for region in regions:
            center_xyz = region.get("center_xyz")
            if center_xyz is None or len(center_xyz) != 3:
                raise ValueError("Cleanup region center must contain exactly three coordinates.")
            half_extent = float(region.get("half_extent_m", 0.0))
            if half_extent <= 0:
                raise ValueError("Cleanup region half extent must be positive.")
            center = np.asarray(center_xyz, dtype=np.float64)
            removed_mask |= np.all(np.abs(xyz - center.reshape(1, 3)) <= half_extent, axis=1)
            normalized_regions.append(
                {
                    "center_xyz": [float(v) for v in center.tolist()],
                    "half_extent_m": float(half_extent),
                    "selection_mode": str(region.get("selection_mode", "unknown")),
                    "updated_at": str(region.get("updated_at", _now())),
                }
            )

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

        selection_mode_summary: dict[str, int] = {}
        for region in normalized_regions:
            mode = region["selection_mode"]
            selection_mode_summary[mode] = selection_mode_summary.get(mode, 0) + 1
        last_region = normalized_regions[-1]

        preview = {
            "scene_key": self.scene_key,
            "selection_mode": last_region["selection_mode"],
            "center_xyz": list(last_region["center_xyz"]),
            "half_extent_m": float(last_region["half_extent_m"]),
            "cleanup_region_count": len(normalized_regions),
            "regions": normalized_regions,
            "selection_mode_summary": selection_mode_summary,
            "source_projection_las": str(self.projection_las),
            "cleaned_las": str(self.cleanup_clean_las_path),
            "raw_overlay": str(self.cleanup_raw_overlay_path),
            "cleanup_overlay": str(self.cleanup_overlay_path),
            "removed_points": int(removed_mask.sum()),
            "kept_points": int(keep_idx.size),
            "fit_rmse_total_px": float(fit_state.get("fit_rmse_total", np.nan)),
            "updated_at": _now(),
        }
        return self._write_cleanup_preview(preview)

    def add_cleanup_region(
        self,
        center_xyz: list[float],
        half_extent_m: float = 1.0,
        selection_mode: str = "unknown",
    ) -> dict[str, Any]:
        if len(center_xyz) != 3:
            raise ValueError("Cleanup region center must contain exactly three coordinates.")
        if half_extent_m <= 0:
            raise ValueError("Cleanup region half extent must be positive.")
        regions = self._current_regions()
        regions.append(
            {
                "region_id": uuid.uuid4().hex[:10],
                "center_xyz": [float(v) for v in center_xyz],
                "half_extent_m": float(half_extent_m),
                "selection_mode": selection_mode,
                "updated_at": _now(),
            }
        )
        return self._apply_cleanup_regions(regions)

    def undo_cleanup_region(self) -> dict[str, Any]:
        regions = self._current_regions()
        if not regions:
            raise ValueError("No cleanup region to undo.")
        regions.pop()
        if not regions:
            self.clear_cleanup_preview()
            return {
                "scene_key": self.scene_key,
                "cleanup_region_count": 0,
                "regions": [],
                "selection_mode_summary": {},
                "source_projection_las": str(self.projection_las) if self.projection_las else None,
                "cleaned_las": None,
                "raw_overlay": None,
                "cleanup_overlay": None,
                "removed_points": 0,
                "kept_points": 0,
                "fit_rmse_total_px": float(self.get_fit_status().get("fit_rmse_total", np.nan)),
                "updated_at": _now(),
            }
        return self._apply_cleanup_regions(regions)

    def remove_cleanup_region(self, region_id: str) -> dict[str, Any]:
        regions = self._current_regions()
        match_idx = next((idx for idx, region in enumerate(regions) if str(region.get("region_id")) == str(region_id)), None)
        if match_idx is None:
            raise IndexError(f"No cleanup region with id {region_id}.")
        regions.pop(match_idx)
        if not regions:
            self.clear_cleanup_preview()
            return {
                "scene_key": self.scene_key,
                "cleanup_region_count": 0,
                "regions": [],
                "cleanup_region_ids": [],
                "selection_mode_summary": {},
                "source_projection_las": str(self.projection_las) if self.projection_las else None,
                "cleaned_las": None,
                "raw_overlay": None,
                "cleanup_overlay": None,
                "removed_points": 0,
                "kept_points": 0,
                "fit_rmse_total_px": float(self.get_fit_status().get("fit_rmse_total", np.nan)),
                "updated_at": _now(),
            }
        return self._apply_cleanup_regions(regions)

    def clear_cleanup_preview(self) -> None:
        if self.cleanup_preview_json_path.exists():
            self.cleanup_preview_json_path.unlink()
        for path in [self.cleanup_raw_overlay_path, self.cleanup_overlay_path, self.cleanup_clean_las_path]:
            if path.exists():
                path.unlink()
        if OCCLUSION_CLEANUP_MANIFEST.exists():
            with OCCLUSION_CLEANUP_MANIFEST.open("r", newline="") as f:
                reader = csv.DictReader(f)
                rows = [dict(r) for r in reader if dict(r).get("scene_key") != self.scene_key]
            self._rewrite_cleanup_manifest(rows)

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
