import csv
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import laspy
import numpy as np

from ihd.annotation_workspace.scene_service import (
    BROWSER_POINT_BUDGET,
    PREPROCESS_ROOT,
    REPO_ROOT,
    _now,
    collection_tag,
    load_gray_preview,
    metrics_from_residual,
    parse_utc_timestamp,
    path_key,
    resolve_lwhsi_file,
    resolve_raw_las,
    resolve_scene_dir,
    resolve_preprocessed_las,
    save_manual_projection_plot,
    save_overlay,
    scene_key,
)
from ihd.datasets.calibration_lidar_cylindrical import calibrate_single, read_corresp, write_cyl
from ihd.datasets.cylindrical_camera import camera, project_vect_safe
from ihd.datasets.depth_rasterization import depth_range, rasterize

WORKSPACE_ROOT_NO_CYL = REPO_ROOT / "analysis" / "annotation_workspace_nocyl"
MIN_NO_CYL_FIT_POINTS = 8


def build_default_cyl_camera(image_width: int, image_height: int) -> tuple[camera, dict[str, Any]]:
    # Conservative generic cylindrical model derived from the observed LWHSI geometry:
    # narrow horizontal angular span, principal row slightly below vertical center,
    # and identity extrinsics as the neutral starting pose.
    horizontal_fov_rad = 1.28
    principal_row_ratio = 0.64875
    focal_height_ratio = 3.6517230769
    radius_guess = 2.5
    principal_angle = 0.0
    pixel_width_angle = horizontal_fov_rad / float(image_width)
    focal_length = focal_height_ratio * float(image_height)
    principal_row = principal_row_ratio * float(image_height)
    rotation = np.eye(3, dtype=np.float64)
    translation = np.zeros(3, dtype=np.float64)
    cam = camera(radius_guess, principal_angle, pixel_width_angle, focal_length, principal_row, rotation, translation)
    return cam, {
        "R": float(radius_guess),
        "w": float(principal_angle),
        "y": float(pixel_width_angle),
        "f": float(focal_length),
        "j0": float(principal_row),
        "Rot": rotation.tolist(),
        "t": translation.tolist(),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "horizontal_fov_rad": float(horizontal_fov_rad),
    }


class NoCylSceneWorkspace:
    def __init__(self, collection: str, path_name: str, step: int | str):
        self.collection = collection
        self.path_name = path_name
        self.step = int(step)
        self.scene_dir = resolve_scene_dir(collection, path_name, self.step)
        self.path_key = path_key(path_name)
        self.scene_key = scene_key(path_name, self.step)
        self.workspace_dir = WORKSPACE_ROOT_NO_CYL / collection / self.path_key / self.scene_key
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        self.raw_las = resolve_raw_las(self.scene_dir, collection, path_name, self.step)
        self.preprocess_out_dir = PREPROCESS_ROOT / collection / self.path_key / self.scene_key
        self.projection_las = self._resolve_preprocessed_las("projection")
        self.hdr_path = resolve_lwhsi_file(self.scene_dir, collection, path_name, self.step, ".hdr", required=True)

        self.scene_json_path = self.workspace_dir / "scene.json"
        self.picks_json_path = self.workspace_dir / "picks.json"
        self.session_json_path = self.workspace_dir / "session.json"
        self.image_preview_path = self.workspace_dir / "image_preview.png"
        self.export_csv_path = self.workspace_dir / "export_picks.csv"
        self.generated_corresp_path = self.workspace_dir / "generated_corresp.txt"
        self.fit_json_path = self.workspace_dir / "fit.json"
        self.fitted_cyl_path = self.workspace_dir / "fitted.cyl"
        self.overlay_preview_path = self.workspace_dir / "overlay_preview.png"
        self.reprojection_preview_path = self.workspace_dir / "reprojection_preview.png"

        self._lock = threading.Lock()
        self._pointcloud_payload: dict[str, Any] | None = None

    def _resolve_preprocessed_las(self, kind: str) -> Path | None:
        try:
            return resolve_preprocessed_las(self.collection, self.path_name, self.step, kind)
        except FileNotFoundError:
            return None

    def preprocessing_ready(self) -> bool:
        self.projection_las = self._resolve_preprocessed_las("projection")
        return self.projection_las is not None

    def get_preprocess_status(self) -> dict[str, Any]:
        ready = self.preprocessing_ready()
        if ready:
            return {
                "state": "ready",
                "ready": True,
                "updated_at": _now(),
                "projection_las": str(self.projection_las),
            }
        return {
            "state": "missing",
            "ready": False,
            "updated_at": _now(),
            "message": "Preprocessing outputs are not available yet.",
        }

    def prepare(self) -> None:
        self._ensure_preview()
        if not self.preprocessing_ready():
            raise FileNotFoundError(
                f"Missing preprocessed display cloud for {self.collection} {self.path_name} step {self.step}: "
                f"{self.preprocess_out_dir}"
            )
        if not self.picks_json_path.exists():
            self._write_json(self.picks_json_path, {"scene_key": self.scene_key, "picks": []})
        if not self.session_json_path.exists():
            self._write_json(
                self.session_json_path,
                {
                    "scene_key": self.scene_key,
                    "selected_target_index": None,
                    "show_targets": True,
                    "show_picked_points": True,
                    "timing_running": False,
                    "timing_started_at": None,
                    "elapsed_seconds": 0.0,
                    "replacement_count": 0,
                    "clear_count": 0,
                    "verdict": None,
                    "updated_at": _now(),
                },
            )
        self._write_generated_corresp_txt()
        self._write_json(self.scene_json_path, self._build_scene_json())

    def _ensure_preview(self) -> None:
        if self.image_preview_path.exists():
            return
        gray8 = load_gray_preview(self.hdr_path)
        import cv2

        if not cv2.imwrite(str(self.image_preview_path), gray8):
            raise RuntimeError(f"Failed to write preview image to {self.image_preview_path}")

    def _build_scene_json(self) -> dict[str, Any]:
        picks = self.get_picks()["picks"]
        gray = load_gray_preview(self.hdr_path)
        default_cam, default_cam_dict = build_default_cyl_camera(gray.shape[1], gray.shape[0])
        _ = default_cam
        return {
            "collection": self.collection,
            "path_name": self.path_name,
            "path_key": self.path_key,
            "step": self.step,
            "scene_key": self.scene_key,
            "scene_label": f"{self.path_name.split('_DistStA')[0]} Step{self.step}",
            "workspace_dir": str(self.workspace_dir),
            "source_paths": {
                "scene_dir": str(self.scene_dir),
                "raw_las": str(self.raw_las),
                "projection_las": str(self.projection_las) if self.projection_las else None,
                "hsi_hdr": str(self.hdr_path),
                "image_preview": str(self.image_preview_path),
                "generated_corresp_txt": str(self.generated_corresp_path),
                "fitted_cyl": str(self.fitted_cyl_path) if self.fitted_cyl_path.exists() else None,
            },
            "capabilities": {
                "has_reference_targets": False,
                "has_reference_cyl": False,
                "can_fit_generated_cyl": True,
                "has_corresp_txt": False,
                "can_run_fit_feedback": False,
            },
            "defaults": {
                "fit_opt_mode": "all",
                "min_fit_points": MIN_NO_CYL_FIT_POINTS,
                "recommended_target_count": [8, 12],
                "default_camera": default_cam_dict,
            },
            "target_count": len(picks),
            "preprocessing": self.get_preprocess_status(),
        }

    def get_scene_payload(self) -> dict[str, Any]:
        scene = self._read_json(self.scene_json_path)
        picks = self.get_picks()["picks"]
        picked_count = sum(1 for p in picks if p["status"] == "picked")
        scene["picks_summary"] = {
            "picked_count": picked_count,
            "missing_count": len(picks) - picked_count,
        }
        return scene

    def get_picks(self) -> dict[str, Any]:
        return self._read_json(self.picks_json_path)

    def get_session(self) -> dict[str, Any]:
        data = self._read_json(self.session_json_path)
        elapsed = float(data.get("elapsed_seconds", 0.0))
        if data.get("timing_running"):
            started = parse_utc_timestamp(data.get("timing_started_at"))
            if started is not None:
                elapsed += max((datetime.now(timezone.utc) - started).total_seconds(), 0.0)
        data["elapsed_seconds_current"] = elapsed
        return data

    def upsert_pick(self, index: int, las_xyz: list[float]) -> dict[str, Any]:
        with self._lock:
            data = self.get_picks()
            picks = data["picks"]
            target = next((p for p in picks if p["index"] == index), None)
            if target is None:
                raise KeyError(f"Unknown target index {index}")
            had_any_picks = any(p.get("las_xyz") is not None for p in picks)
            replaced = target.get("las_xyz") is not None
            target["las_xyz"] = [float(v) for v in las_xyz]
            target["status"] = "picked"
            target["updated_at"] = _now()
            self._write_json(self.picks_json_path, data)
            self._write_generated_corresp_txt(data["picks"])
            session = self._read_json(self.session_json_path)
            if (not had_any_picks) and (not session.get("timing_running")) and float(session.get("elapsed_seconds", 0.0)) <= 0.0:
                session["timing_running"] = True
                session["timing_started_at"] = _now()
            if replaced:
                session["replacement_count"] = int(session.get("replacement_count", 0)) + 1
            session["updated_at"] = _now()
            self._write_json(self.session_json_path, session)
            return target

    def clear_pick(self, index: int) -> dict[str, Any]:
        with self._lock:
            data = self.get_picks()
            picks = data["picks"]
            target = next((p for p in picks if p["index"] == index), None)
            if target is None:
                raise KeyError(f"Unknown target index {index}")
            target["las_xyz"] = None
            target["status"] = "empty"
            target["updated_at"] = _now()
            self._write_json(self.picks_json_path, data)
            self._write_generated_corresp_txt(data["picks"])
            session = self._read_json(self.session_json_path)
            session["clear_count"] = int(session.get("clear_count", 0)) + 1
            session["updated_at"] = _now()
            self._write_json(self.session_json_path, session)
            return target

    def add_target(self, image_uv: list[float]) -> dict[str, Any]:
        with self._lock:
            data = self.get_picks()
            picks = data["picks"]
            next_index = max((p["index"] for p in picks), default=-1) + 1
            target = {
                "index": int(next_index),
                "image_uv": [float(image_uv[0]), float(image_uv[1])],
                "las_xyz": None,
                "status": "empty",
                "created_at": _now(),
                "updated_at": _now(),
            }
            picks.append(target)
            self._write_json(self.picks_json_path, data)
            self._write_generated_corresp_txt(data["picks"])
            return target

    def delete_target(self, index: int) -> dict[str, Any]:
        with self._lock:
            data = self.get_picks()
            picks = data["picks"]
            target = next((p for p in picks if p["index"] == index), None)
            if target is None:
                raise KeyError(f"Unknown target index {index}")
            data["picks"] = [p for p in picks if p["index"] != index]
            self._write_json(self.picks_json_path, data)
            self._write_generated_corresp_txt(data["picks"])
            session = self._read_json(self.session_json_path)
            if session.get("selected_target_index") == index:
                session["selected_target_index"] = None
            session["updated_at"] = _now()
            self._write_json(self.session_json_path, session)
            return target

    def update_session(self, selected_target_index: int | None) -> dict[str, Any]:
        data = self._read_json(self.session_json_path)
        data["selected_target_index"] = selected_target_index
        data["updated_at"] = _now()
        self._write_json(self.session_json_path, data)
        return self.get_session()

    def start_timer(self) -> dict[str, Any]:
        data = self._read_json(self.session_json_path)
        if not data.get("timing_running"):
            data["timing_running"] = True
            data["timing_started_at"] = _now()
            data["updated_at"] = _now()
            self._write_json(self.session_json_path, data)
        return self.get_session()

    def stop_timer(self, elapsed_seconds_override: float | None = None) -> dict[str, Any]:
        data = self._read_json(self.session_json_path)
        if data.get("timing_running"):
            if elapsed_seconds_override is not None:
                elapsed = max(float(elapsed_seconds_override), 0.0)
            else:
                elapsed = float(data.get("elapsed_seconds", 0.0))
                started = parse_utc_timestamp(data.get("timing_started_at"))
                if started is not None:
                    elapsed += max((datetime.now(timezone.utc) - started).total_seconds(), 0.0)
            data["elapsed_seconds"] = elapsed
            data["timing_running"] = False
            data["timing_started_at"] = None
            data["updated_at"] = _now()
            self._write_json(self.session_json_path, data)
        return self.get_session()

    def reset_timer(self) -> dict[str, Any]:
        data = self._read_json(self.session_json_path)
        data["elapsed_seconds"] = 0.0
        data["timing_running"] = False
        data["timing_started_at"] = None
        data["updated_at"] = _now()
        self._write_json(self.session_json_path, data)
        return self.get_session()

    def set_verdict(self, verdict: str | None) -> dict[str, Any]:
        allowed = {None, "good", "usable with caution", "bad"}
        if verdict not in allowed:
            raise ValueError(f"Unsupported verdict: {verdict}")
        data = self._read_json(self.session_json_path)
        if data.get("timing_running"):
            data = self.stop_timer()
            data = dict(data)
        data["verdict"] = verdict
        data["updated_at"] = _now()
        self._write_json(self.session_json_path, data)
        return self.get_session()

    def export_picks_csv(self) -> Path:
        data = self.get_picks()
        with self.export_csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "image_u", "image_v", "las_x", "las_y", "las_z", "status"])
            for p in data["picks"]:
                las_xyz = p["las_xyz"] or [None, None, None]
                writer.writerow([p["index"], p["image_uv"][0], p["image_uv"][1], las_xyz[0], las_xyz[1], las_xyz[2], p["status"]])
        return self.export_csv_path

    def _write_generated_corresp_txt(self, picks: list[dict[str, Any]] | None = None) -> Path:
        rows = []
        for pick in picks if picks is not None else self.get_picks()["picks"]:
            image_uv = pick.get("image_uv")
            las_xyz = pick.get("las_xyz")
            if image_uv is None or las_xyz is None:
                continue
            rows.append((int(pick["index"]), float(image_uv[0]), float(image_uv[1]), float(las_xyz[0]), float(las_xyz[1]), float(las_xyz[2])))
        rows.sort(key=lambda row: row[0])
        with self.generated_corresp_path.open("w", newline="") as f:
            f.write("i j X Y Z\n")
            for _, i_val, j_val, x_val, y_val, z_val in rows:
                f.write(f"{i_val:.6f} {j_val:.6f} {x_val:.9f} {y_val:.9f} {z_val:.9f}\n")
        return self.generated_corresp_path

    def export_generated_corresp_txt(self) -> Path:
        return self._write_generated_corresp_txt()

    def get_fit_status(self) -> dict[str, Any]:
        if self.fit_json_path.exists():
            return self._read_json(self.fit_json_path)
        gray = load_gray_preview(self.hdr_path)
        _, default_cam_dict = build_default_cyl_camera(gray.shape[1], gray.shape[0])
        generated_count = sum(1 for p in self.get_picks()["picks"] if p.get("las_xyz") is not None and p.get("image_uv") is not None)
        return {
            "state": "idle",
            "ready": False,
            "mode": "generated_cyl",
            "picked_generated_points": int(generated_count),
            "min_fit_points": MIN_NO_CYL_FIT_POINTS,
            "can_fit_generated_cyl": True,
            "overlay_preview": None,
            "reprojection_preview": None,
            "fitted_cyl": str(self.fitted_cyl_path) if self.fitted_cyl_path.exists() else None,
            "fit_reference_uv": [],
            "fit_projected_uv": [],
            "default_camera": default_cam_dict,
        }

    def get_pointcloud_payload(self) -> dict[str, Any]:
        if not self.preprocessing_ready() or self.projection_las is None:
            raise FileNotFoundError("Preprocessing is not ready yet for this scene.")
        if self._pointcloud_payload is None:
            las = laspy.read(self.projection_las)
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
            self._pointcloud_payload = {
                "source_kind": "display",
                "display_source": "projection_las",
                "source_las": str(self.projection_las),
                "point_count": int(xyz.shape[0]),
                "original_point_count": original_point_count,
                "browser_point_budget": BROWSER_POINT_BUDGET,
                "x": xyz[:, 0].tolist(),
                "y": xyz[:, 1].tolist(),
                "z": xyz[:, 2].tolist(),
                "intensity_norm": intensity_norm.tolist(),
            }
        return self._pointcloud_payload

    def compute_fit(self) -> dict[str, Any]:
        self._write_generated_corresp_txt()
        corr_i, corr_j, corr_xyz = read_corresp(str(self.generated_corresp_path))
        if corr_xyz.shape[0] < MIN_NO_CYL_FIT_POINTS:
            gray = load_gray_preview(self.hdr_path)
            _, default_cam_dict = build_default_cyl_camera(gray.shape[1], gray.shape[0])
            status = {
                "state": "insufficient_points",
                "ready": False,
                "mode": "generated_cyl",
                "picked_generated_points": int(corr_xyz.shape[0]),
                "min_fit_points": MIN_NO_CYL_FIT_POINTS,
                "can_fit_generated_cyl": True,
                "overlay_preview": None,
                "reprojection_preview": None,
                "fitted_cyl": None,
                "fit_reference_uv": [],
                "fit_projected_uv": [],
                "default_camera": default_cam_dict,
            }
            self._write_json(self.fit_json_path, status)
            return status

        gray = load_gray_preview(self.hdr_path).astype(np.float64) / 255.0
        H, W = gray.shape
        cam_init, default_cam_dict = build_default_cyl_camera(W, H)
        cam_opt, stats = calibrate_single(
            corr_i,
            corr_j,
            corr_xyz,
            cam_init,
            opt_mode="all",
            image_width=W,
            image_height=H,
            max_iters=400,
        )
        write_cyl(cam_opt, str(self.fitted_cyl_path))

        uv_fit = project_vect_safe(corr_xyz, cam_opt)
        residual = uv_fit - np.column_stack((corr_i, corr_j))
        fit_metrics = metrics_from_residual(residual)
        save_manual_projection_plot(gray, np.column_stack((corr_i, corr_j)), uv_fit, self.reprojection_preview_path, f"{self.path_name.split('_DistStA')[0]} Step{self.step}")

        las = laspy.read(self.projection_las)
        xyz_las = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
        Pc = (cam_opt.Rot @ xyz_las.T).T + cam_opt.t.reshape(1, 3)
        d = depth_range(Pc)
        ij = project_vect_safe(xyz_las, cam_opt)
        i_vals = ij[:, 0]
        j_vals = ij[:, 1]
        valid = np.isfinite(i_vals) & np.isfinite(j_vals) & np.isfinite(d)
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
        save_overlay(
            gray,
            depth_img,
            self.overlay_preview_path,
            f"{self.path_name.split('_DistStA')[0]} Step{self.step}",
        )

        status = {
            "state": "ready",
            "ready": True,
            "mode": "generated_cyl",
            "updated_at": _now(),
            "picked_generated_points": int(corr_xyz.shape[0]),
            "min_fit_points": MIN_NO_CYL_FIT_POINTS,
            "fit_rmse_total": fit_metrics["rmse_total"],
            "fit_rmse_u": fit_metrics["rmse_u"],
            "fit_rmse_v": fit_metrics["rmse_v"],
            "fit_mean_abs_du": fit_metrics["mean_abs_du"],
            "fit_mean_abs_dv": fit_metrics["mean_abs_dv"],
            "optimizer_stats": stats,
            "overlay_preview": str(self.overlay_preview_path),
            "reprojection_preview": str(self.reprojection_preview_path),
            "fitted_cyl": str(self.fitted_cyl_path),
            "fit_reference_uv": np.column_stack((corr_i, corr_j)).tolist(),
            "fit_projected_uv": uv_fit.tolist(),
            "default_camera": default_cam_dict,
            "optimized_camera": {
                "R": float(cam_opt.R),
                "w": float(cam_opt.w),
                "y": float(cam_opt.y),
                "f": float(cam_opt.f),
                "j0": float(cam_opt.j0),
                "Rot": cam_opt.Rot.tolist(),
                "t": cam_opt.t.tolist(),
            },
        }
        self._write_json(self.fit_json_path, status)
        return status

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text())

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
