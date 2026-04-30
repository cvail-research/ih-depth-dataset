import csv
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spectral as spy

from ihd.datasets.calibration_lidar_cylindrical import calibrate_single, read_corresp, write_cyl
from ihd.datasets.cylindrical_camera import project_vect_safe, read_cam
from ihd.datasets.depth_rasterization import depth_range, rasterize

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = REPO_ROOT / "analysis"
WORKSPACE_ROOT = ANALYSIS_ROOT / "annotation_workspace"
PREPROCESS_ROOT = ANALYSIS_ROOT / "lidar_preprocessing"
BROWSER_POINT_BUDGET = 260_000
MIN_FIT_POINTS = 6


def collection_tag(collection: str) -> str:
    return collection.replace("_DistStA", "")


def path_key(path_name: str) -> str:
    return path_name.split("_DistStA")[0].lower()


def scene_key(path_name: str, step: int | str) -> str:
    return f"{path_key(path_name)}_step{int(step)}"


def parse_utc_timestamp(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def resolve_scene_dir(collection: str, path_name: str, step: int | str) -> Path:
    prefix = path_name.split("_DistStA")[0]
    step = int(step)
    candidates = [
        Path("/disk") / collection / path_name / f"{prefix}_Step{step}_DistStA",
        Path("/disk") / collection / path_name / f"{prefix}_Step{step}",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not resolve scene directory for {collection} {path_name} step {step}")


def resolve_lwhsi_file(scene_dir: Path, collection: str, path_name: str, step: int | str, suffix: str, required: bool) -> Path | None:
    prefix = path_name.split("_DistStA")[0]
    step = int(step)
    candidates = list(scene_dir.glob(f"*{prefix}_Step{step}*LWHSI1*{suffix}"))
    if not candidates:
        candidates = list(scene_dir.glob(f"*LWHSI1*{suffix}"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if required:
        raise FileNotFoundError(f"Could not resolve LWHSI file for suffix {suffix} in {scene_dir}")
    return None


def resolve_raw_las(scene_dir: Path, collection: str, path_name: str, step: int | str) -> Path:
    prefix = path_name.split("_DistStA")[0]
    step = int(step)
    candidates = list(scene_dir.glob(f"*{prefix}_Step{step}*HiResLIDAR*.las"))
    if not candidates:
        candidates = list(scene_dir.glob("*HiResLIDAR*.las"))
    if not candidates:
        raise FileNotFoundError(f"Could not resolve HiResLIDAR LAS in {scene_dir}")
    candidates.sort()
    return candidates[0]


def resolve_preprocessed_las(collection: str, path_name: str, step: int | str, kind: str) -> Path:
    prefix = path_name.split("_DistStA")[0]
    step = int(step)
    pre_dir = PREPROCESS_ROOT / collection / path_key(path_name) / scene_key(path_name, step)
    candidates = list(pre_dir.glob(f"*{prefix}_Step{step}*HiResLIDAR*_{kind}_clean.las"))
    if not candidates:
        candidates = list(pre_dir.glob(f"*HiResLIDAR*_{kind}_clean.las"))
    if not candidates:
        raise FileNotFoundError(
            f"Missing preprocessed {kind} LAS for {collection} {path_name} step {step}: {pre_dir}"
        )
    candidates.sort()
    return candidates[0]


def load_gray_preview(hdr_path: Path) -> np.ndarray:
    bsq_path = hdr_path.with_suffix(".bsq")
    img = spy.envi.open(str(hdr_path), str(bsq_path))
    cube = img.load()
    gray = cube.sum(axis=-1).astype(np.float64)
    gray -= gray.min()
    if gray.max() > 0:
        gray /= gray.max()
    gray8 = np.clip(np.round(gray * 255.0), 0, 255).astype(np.uint8)
    return gray8


def read_corresp_txt(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Empty correspondence file: {path}")

    def is_numeric(tok: str) -> bool:
        try:
            float(tok)
            return True
        except ValueError:
            return False

    if lines and not is_numeric(lines[0].split()[0]) and len(lines[0].split()) >= 5:
        lines = lines[1:]

    if len(lines[0].split()) == 1 and lines[0].split()[0].isdigit():
        count = int(lines[0].split()[0])
        lines = lines[1 : 1 + count]

    vals = []
    for line in lines:
        parts = line.split()
        if len(parts) == 5 and all(is_numeric(p) for p in parts):
            vals.append([float(p) for p in parts])

    arr = np.asarray(vals, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 5:
        raise ValueError(f"Invalid correspondence file format: {path}")
    return arr[:, 0], arr[:, 1], arr[:, 2:5]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fit_rigid_transform(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    H = src_centered.T @ dst_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = dst_mean - (R @ src_mean)
    return R, t


def project_manual_points(
    las_points: np.ndarray,
    uv_gt: np.ndarray,
    cam,
    R_las_to_txt: np.ndarray,
    t_txt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts_txt = (R_las_to_txt @ las_points.T).T + t_txt.reshape(1, 3)
    uv_pred = project_vect_safe(pts_txt, cam)
    residual = uv_pred - uv_gt
    return pts_txt, uv_pred, residual


def metrics_from_residual(residual: np.ndarray) -> dict[str, float]:
    return {
        "rmse_u": float(np.sqrt(np.mean(residual[:, 0] ** 2))),
        "rmse_v": float(np.sqrt(np.mean(residual[:, 1] ** 2))),
        "rmse_total": float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))),
        "mean_abs_du": float(np.mean(np.abs(residual[:, 0]))),
        "mean_abs_dv": float(np.mean(np.abs(residual[:, 1]))),
    }


def save_manual_projection_plot(
    gray: np.ndarray,
    uv_gt: np.ndarray,
    uv_fit: np.ndarray,
    out_path: Path,
    scene_label: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 4), dpi=160)
    ax.imshow(gray, cmap="gray", interpolation="nearest")
    ax.scatter(uv_gt[:, 0], uv_gt[:, 1], s=44, facecolors="none", edgecolors="red", linewidths=1.5, label="corr 2D")
    ax.scatter(uv_fit[:, 0], uv_fit[:, 1], s=28, marker="+", c="yellow", linewidths=1.2, label="rigid fit")
    for gt, pred in zip(uv_gt, uv_fit):
        ax.plot([gt[0], pred[0]], [gt[1], pred[1]], color="yellow", alpha=0.5, linewidth=0.8)
    ax.set_xlim(0, gray.shape[1] - 1)
    ax.set_ylim(gray.shape[0] - 1, 0)
    ax.set_title(f"{scene_label}: reprojection preview")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_cyl_verification_plot(
    gray: np.ndarray,
    uv_gt: np.ndarray,
    uv_cyl: np.ndarray,
    out_path: Path,
    scene_label: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 4), dpi=160)
    ax.imshow(gray, cmap="gray", interpolation="nearest")
    ax.scatter(uv_gt[:, 0], uv_gt[:, 1], s=44, facecolors="none", edgecolors="red", linewidths=1.5, label="corr 2D")
    ax.scatter(uv_cyl[:, 0], uv_cyl[:, 1], s=28, marker="x", c="cyan", linewidths=1.2, label="txt XYZ via .cyl")
    for gt, pred in zip(uv_gt, uv_cyl):
        ax.plot([gt[0], pred[0]], [gt[1], pred[1]], color="yellow", alpha=0.5, linewidth=0.8)
    ax.set_xlim(0, gray.shape[1] - 1)
    ax.set_ylim(gray.shape[0] - 1, 0)
    ax.set_title(f"{scene_label}: .cyl verification")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_overlay(gray: np.ndarray, depth_img: np.ndarray, out_path: Path, scene_label: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask_valid = np.isfinite(depth_img)
    if not np.any(mask_valid):
        raise ValueError("No valid depth pixels for overlay.")
    d_min = float(np.nanmin(depth_img[mask_valid]))
    d_max = float(np.nanmax(depth_img[mask_valid]))
    if d_max <= d_min:
        d_max = d_min + 1e-6
    H, W = gray.shape
    dpi = 100
    cb_px = 22
    title_px = 18
    total_h = H + cb_px + title_px
    fig = plt.figure(figsize=(W / dpi, total_h / dpi), dpi=dpi)
    ax_title = fig.add_axes([0.0, (H + cb_px) / total_h, 1.0, title_px / total_h])
    ax_img = fig.add_axes([0.0, 0.0, 1.0, H / total_h])
    ax_cb = fig.add_axes([0.0, H / total_h, 1.0, cb_px / total_h])
    ax_title.axis("off")
    ax_title.text(0.5, 0.5, scene_label, ha="center", va="center", fontsize=9)
    ax_img.imshow(gray, cmap="gray", interpolation="nearest")
    yv, xv = np.nonzero(mask_valid)
    ax_img.scatter(xv, yv, c=depth_img[mask_valid], s=1, cmap="viridis_r", vmin=d_min, vmax=d_max, marker="s", linewidths=0)
    ax_img.set_xlim(0, W)
    ax_img.set_ylim(H, 0)
    ax_img.axis("off")
    gradient = np.linspace(d_min, d_max, max(2, W), dtype=np.float32)[None, :]
    ax_cb.imshow(gradient, aspect="auto", cmap="viridis_r", vmin=d_min, vmax=d_max, extent=[d_min, d_max, 0, 1])
    ax_cb.set_xlim(d_min, d_max)
    ax_cb.set_yticks([])
    span = d_max - d_min
    inset = max(span * 0.015, 1e-6)
    ax_cb.set_xticks([d_min + inset, d_max - inset])
    labels = ax_cb.set_xticklabels([f"{int(round(d_min))} m", f"{int(round(d_max))} m"])
    if len(labels) == 2:
        labels[0].set_ha("left")
        labels[1].set_ha("right")
    ax_cb.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, length=0, pad=2, labelsize=8)
    for spine in ax_cb.spines.values():
        spine.set_visible(False)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


class SceneWorkspace:
    def __init__(
        self,
        collection: str,
        path_name: str,
        step: int | str,
        *,
        force_generated_cyl_mode: bool = False,
        use_reference_targets_in_generated_mode: bool = False,
        workspace_variant: str | None = None,
        default_init_cyl_path: str | None = None,
        default_fit_opt_mode: str | None = None,
    ):
        self.collection = collection
        self.path_name = path_name
        self.step = int(step)
        self.force_generated_cyl_mode = force_generated_cyl_mode
        self.use_reference_targets_in_generated_mode = use_reference_targets_in_generated_mode
        self.workspace_variant = workspace_variant.strip() if workspace_variant else None
        self.default_init_cyl_path = str(Path(default_init_cyl_path).expanduser()) if default_init_cyl_path else None
        self.default_fit_opt_mode = (default_fit_opt_mode or "all").strip()
        self.scene_dir = resolve_scene_dir(collection, path_name, self.step)
        self.base_scene_key = scene_key(path_name, self.step)
        self.scene_key = self.base_scene_key if not self.workspace_variant else f"{self.base_scene_key}__{self.workspace_variant}"
        self.path_key = path_key(path_name)
        self.workspace_dir = WORKSPACE_ROOT / collection / self.path_key / self.scene_key
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        self.raw_las = resolve_raw_las(self.scene_dir, collection, path_name, self.step)
        self.preprocess_out_dir = PREPROCESS_ROOT / collection / self.path_key / self.scene_key
        self.projection_las = self._resolve_preprocessed_las("projection")
        self.hdr_path = resolve_lwhsi_file(self.scene_dir, collection, path_name, self.step, ".hdr", required=True)
        self.reference_cyl_path = resolve_lwhsi_file(self.scene_dir, collection, path_name, self.step, ".cyl", required=False)
        self.reference_corresp_path = resolve_lwhsi_file(self.scene_dir, collection, path_name, self.step, "_corresp.txt", required=False)
        self.cyl_path = None if self.force_generated_cyl_mode else self.reference_cyl_path
        if self.force_generated_cyl_mode and self.use_reference_targets_in_generated_mode:
            self.corresp_path = self.reference_corresp_path
        else:
            self.corresp_path = None if self.force_generated_cyl_mode else self.reference_corresp_path

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
        self.cyl_verification_overlay_path = self.workspace_dir / "cyl_verification_overlay.png"

        self._lock = threading.Lock()
        self._pointcloud_payload: dict[str, Any] | None = None

    def prepare(self) -> None:
        self._ensure_preview()
        if not self.preprocessing_ready():
            raise FileNotFoundError(
                f"Missing preprocessed display cloud for {self.collection} {self.path_name} step {self.step}: "
                f"{self.preprocess_out_dir}"
            )
        picks = self._initial_picks()
        if not self.picks_json_path.exists():
            self._write_json(self.picks_json_path, {"scene_key": self.scene_key, "picks": picks})
        elif self.force_generated_cyl_mode and self.use_reference_targets_in_generated_mode:
            existing = self._read_json(self.picks_json_path)
            if not self._picks_match_reference_targets(existing.get("picks", []), picks):
                self._write_json(self.picks_json_path, {"scene_key": self.scene_key, "picks": picks})
                if self.fit_json_path.exists():
                    self.fit_json_path.unlink()
        if not self.session_json_path.exists():
            selected = picks[0]["index"] if picks else None
            self._write_json(
                self.session_json_path,
                {
                    "scene_key": self.scene_key,
                    "selected_target_index": selected,
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

    @staticmethod
    def _picks_match_reference_targets(existing: list[dict[str, Any]], reference: list[dict[str, Any]]) -> bool:
        if len(existing) != len(reference):
            return False
        for lhs, rhs in zip(existing, reference):
            if int(lhs.get("index", -1)) != int(rhs.get("index", -1)):
                return False
            lhs_uv = lhs.get("image_uv")
            rhs_uv = rhs.get("image_uv")
            if lhs_uv is None or rhs_uv is None or len(lhs_uv) != 2 or len(rhs_uv) != 2:
                return False
            if abs(float(lhs_uv[0]) - float(rhs_uv[0])) > 1e-6 or abs(float(lhs_uv[1]) - float(rhs_uv[1])) > 1e-6:
                return False
        return True

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

    def _initial_picks(self) -> list[dict[str, Any]]:
        if not self.corresp_path:
            return []
        corr_i, corr_j, corr_xyz = read_corresp_txt(self.corresp_path)
        picks = []
        for idx, (u, v, xyz) in enumerate(zip(corr_i.tolist(), corr_j.tolist(), corr_xyz.tolist())):
            picks.append(
                {
                    "index": idx,
                    "image_uv": [float(u), float(v)],
                    "txt_xyz": None if self.force_generated_cyl_mode else [float(x) for x in xyz],
                    "las_xyz": None,
                    "status": "empty",
                    "created_at": _now(),
                    "updated_at": _now(),
                }
            )
        return picks

    def _build_scene_json(self) -> dict[str, Any]:
        picks = self.get_picks()["picks"]
        preprocess_status = self.get_preprocess_status()
        return {
            "collection": self.collection,
            "path_name": self.path_name,
            "path_key": self.path_key,
            "step": self.step,
            "scene_key": self.scene_key,
            "base_scene_key": self.base_scene_key,
            "scene_label": f"{self.path_name.split('_DistStA')[0]} Step{self.step}",
            "workspace_variant": self.workspace_variant,
            "workspace_dir": str(self.workspace_dir),
            "source_paths": {
                "scene_dir": str(self.scene_dir),
                "raw_las": str(self.raw_las),
                "projection_las": str(self.projection_las) if self.projection_las else None,
                "hsi_hdr": str(self.hdr_path),
                "image_preview": str(self.image_preview_path),
                "generated_corresp_txt": str(self.generated_corresp_path),
                "cyl": str(self.cyl_path) if self.cyl_path else None,
                "corresp_txt": str(self.corresp_path) if self.corresp_path else None,
                "reference_cyl": str(self.reference_cyl_path) if self.reference_cyl_path else None,
                "reference_corresp_txt": str(self.reference_corresp_path) if self.reference_corresp_path else None,
                "default_init_cyl": self.default_init_cyl_path,
                "default_fit_opt_mode": self.default_fit_opt_mode,
            },
            "capabilities": {
                "has_cyl": self.cyl_path is not None,
                "has_corresp_txt": self.corresp_path is not None,
                "can_run_fit_feedback": self.cyl_path is not None and self.corresp_path is not None,
                "can_fit_generated_cyl": True,
                "force_generated_cyl_mode": self.force_generated_cyl_mode,
                "use_reference_targets_in_generated_mode": self.use_reference_targets_in_generated_mode,
            },
            "target_count": len(picks),
            "preprocessing": preprocess_status,
        }

    def _ensure_preview(self) -> None:
        if self.image_preview_path.exists():
            return
        gray8 = load_gray_preview(self.hdr_path)
        if not cv2.imwrite(str(self.image_preview_path), gray8):
            raise RuntimeError(f"Failed to write preview image to {self.image_preview_path}")

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
                "txt_xyz": None,
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
        # Case 1 uses fixed reference targets, so deleting from the image means
        # clearing the LiDAR pick for that target instead of removing the target.
        return self.clear_pick(index)

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
            writer.writerow(
                [
                    "index",
                    "image_u",
                    "image_v",
                    "txt_x",
                    "txt_y",
                    "txt_z",
                    "las_x",
                    "las_y",
                    "las_z",
                    "status",
                ]
            )
            for p in data["picks"]:
                txt_xyz = p["txt_xyz"] or [None, None, None]
                las_xyz = p["las_xyz"] or [None, None, None]
                writer.writerow(
                    [
                        p["index"],
                        p["image_uv"][0],
                        p["image_uv"][1],
                        txt_xyz[0],
                        txt_xyz[1],
                        txt_xyz[2],
                        las_xyz[0],
                        las_xyz[1],
                        las_xyz[2],
                        p["status"],
                    ]
                )
        return self.export_csv_path

    def _write_generated_corresp_txt(self, picks: list[dict[str, Any]] | None = None) -> Path:
        rows = []
        for pick in picks if picks is not None else self.get_picks()["picks"]:
            image_uv = pick.get("image_uv")
            las_xyz = pick.get("las_xyz")
            if image_uv is None or las_xyz is None:
                continue
            rows.append(
                (
                    int(pick["index"]),
                    float(image_uv[0]),
                    float(image_uv[1]),
                    float(las_xyz[0]),
                    float(las_xyz[1]),
                    float(las_xyz[2]),
                )
            )
        rows.sort(key=lambda row: row[0])
        with self.generated_corresp_path.open("w", newline="") as f:
            f.write("i j X Y Z\n")
            for _, i_val, j_val, x_val, y_val, z_val in rows:
                f.write(f"{i_val:.6f} {j_val:.6f} {x_val:.9f} {y_val:.9f} {z_val:.9f}\n")
        return self.generated_corresp_path

    def export_generated_corresp_txt(self) -> Path:
        return self._write_generated_corresp_txt()

    def _picked_correspondence_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        picked = [
            p for p in self.get_picks()["picks"]
            if p.get("las_xyz") is not None and p.get("txt_xyz") is not None
        ]
        if not picked:
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
            )
        uv = np.asarray([p["image_uv"] for p in picked], dtype=np.float64)
        txt_xyz = np.asarray([p["txt_xyz"] for p in picked], dtype=np.float64)
        las_xyz = np.asarray([p["las_xyz"] for p in picked], dtype=np.float64)
        return uv, txt_xyz, las_xyz

    def get_fit_status(self) -> dict[str, Any]:
        if self.fit_json_path.exists():
            return self._read_json(self.fit_json_path)
        _, txt_xyz, _ = self._picked_correspondence_arrays()
        generated_count = sum(1 for p in self.get_picks()["picks"] if p.get("las_xyz") is not None and p.get("image_uv") is not None)
        return {
            "state": "idle",
            "ready": False,
            "mode": "existing_cyl" if self.cyl_path and self.corresp_path else "generated_cyl",
            "picked_fit_points": int(txt_xyz.shape[0]),
            "picked_generated_points": int(generated_count),
            "min_fit_points": MIN_FIT_POINTS,
            "can_run_fit_feedback": bool(self.cyl_path and self.corresp_path),
            "can_fit_generated_cyl": True,
            "overlay_preview": None,
            "reprojection_preview": None,
            "cyl_verification_overlay": None,
            "fitted_cyl": str(self.fitted_cyl_path) if self.fitted_cyl_path.exists() else None,
            "init_cyl": None,
            "fit_reference_uv": [],
            "fit_projected_uv": [],
        }

    def get_pointcloud_payload(self) -> dict[str, Any]:
        if not self.preprocessing_ready() or self.projection_las is None:
            raise FileNotFoundError("Preprocessing is not ready yet for this scene.")
        if self._pointcloud_payload is None:
            las_path = self.projection_las
            las = laspy.read(las_path)
            xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float32)
            if hasattr(las, "intensity"):
                intensity = np.asarray(las.intensity).astype(np.float32)
                if intensity.size == xyz.shape[0]:
                    finite = np.isfinite(intensity)
                    if np.any(finite):
                        lo = float(np.min(intensity[finite]))
                        hi = float(np.max(intensity[finite]))
                        if hi > lo:
                            intensity_norm = (intensity - lo) / (hi - lo)
                        else:
                            intensity_norm = np.zeros_like(intensity)
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
                "source_las": str(las_path),
                "point_count": int(xyz.shape[0]),
                "original_point_count": original_point_count,
                "browser_point_budget": BROWSER_POINT_BUDGET,
                "x": xyz[:, 0].tolist(),
                "y": xyz[:, 1].tolist(),
                "z": xyz[:, 2].tolist(),
                "intensity_norm": intensity_norm.tolist(),
            }
        return self._pointcloud_payload

    def _compute_existing_cyl_fit(self) -> dict[str, Any]:
        if not (self.cyl_path and self.corresp_path):
            raise ValueError("Fit feedback requires both .cyl and correspondence .txt.")
        uv_gt, corr_xyz, manual_las = self._picked_correspondence_arrays()
        if corr_xyz.shape[0] < MIN_FIT_POINTS:
            status = {
            "state": "insufficient_points",
            "ready": False,
            "mode": "existing_cyl",
            "picked_fit_points": int(corr_xyz.shape[0]),
            "min_fit_points": MIN_FIT_POINTS,
            "can_run_fit_feedback": True,
            "can_fit_generated_cyl": True,
            "overlay_preview": None,
            "reprojection_preview": None,
            "cyl_verification_overlay": None,
            "fitted_cyl": None,
            "init_cyl": str(self.cyl_path),
        }
            self._write_json(self.fit_json_path, status)
            return status

        gray = load_gray_preview(self.hdr_path).astype(np.float64) / 255.0
        cam = read_cam(str(self.cyl_path))
        R_fit, t_fit = fit_rigid_transform(corr_xyz, manual_las)
        R_fit_inv = R_fit.T
        t_fit_inv = -(R_fit_inv @ t_fit)

        uv_cyl = project_vect_safe(corr_xyz, cam)
        res_cyl = uv_cyl - uv_gt
        _, uv_fit, res_fit = project_manual_points(manual_las, uv_gt, cam, R_fit_inv, t_fit_inv)
        fit_metrics = metrics_from_residual(res_fit)
        cyl_metrics = metrics_from_residual(res_cyl)

        save_manual_projection_plot(gray, uv_gt, uv_fit, self.reprojection_preview_path, f"{self.path_name.split('_DistStA')[0]} Step{self.step}")
        save_cyl_verification_plot(gray, uv_gt, uv_cyl, self.cyl_verification_overlay_path, f"{self.path_name.split('_DistStA')[0]} Step{self.step}")

        las = laspy.read(self.projection_las)
        xyz_las = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
        xyz_txt = (R_fit_inv @ xyz_las.T).T + t_fit_inv.reshape(1, 3)
        Pc = (cam.Rot @ xyz_txt.T).T + cam.t.reshape(1, 3)
        d = depth_range(Pc)
        ij = project_vect_safe(xyz_txt, cam)
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
        save_overlay(
            gray,
            depth_img,
            self.overlay_preview_path,
            f"{self.path_name.split('_DistStA')[0]} Step{self.step}",
        )

        status = {
            "state": "ready",
            "ready": True,
            "mode": "existing_cyl",
            "updated_at": _now(),
            "picked_fit_points": int(corr_xyz.shape[0]),
            "min_fit_points": MIN_FIT_POINTS,
            "fit_rmse_total": fit_metrics["rmse_total"],
            "fit_rmse_u": fit_metrics["rmse_u"],
            "fit_rmse_v": fit_metrics["rmse_v"],
            "fit_mean_abs_du": fit_metrics["mean_abs_du"],
            "fit_mean_abs_dv": fit_metrics["mean_abs_dv"],
            "cyl_verify_rmse_total": cyl_metrics["rmse_total"],
            "cyl_verify_rmse_u": cyl_metrics["rmse_u"],
            "cyl_verify_rmse_v": cyl_metrics["rmse_v"],
            "fitted_txt_to_las_rotation": R_fit.tolist(),
            "fitted_t_las": t_fit.tolist(),
            "las_to_txt_rotation": R_fit_inv.tolist(),
            "las_to_txt_translation": t_fit_inv.tolist(),
            "overlay_preview": str(self.overlay_preview_path),
            "reprojection_preview": str(self.reprojection_preview_path),
            "cyl_verification_overlay": str(self.cyl_verification_overlay_path),
            "fitted_cyl": str(self.cyl_path),
            "init_cyl": str(self.cyl_path),
            "fit_reference_uv": uv_gt.tolist(),
            "fit_projected_uv": uv_fit.tolist(),
        }
        self._write_json(self.fit_json_path, status)
        return status

    def _picked_generated_correspondence_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        picked = [
            p for p in self.get_picks()["picks"]
            if p.get("las_xyz") is not None and p.get("image_uv") is not None
        ]
        if not picked:
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
            )
        picked.sort(key=lambda p: p["index"])
        uv = np.asarray([p["image_uv"] for p in picked], dtype=np.float64)
        las_xyz = np.asarray([p["las_xyz"] for p in picked], dtype=np.float64)
        return uv, las_xyz

    def _picked_reference_target_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.reference_corresp_path:
            raise ValueError("Reference correspondence file is required for generated recovery mode.")
        corr_i, corr_j, corr_xyz = read_corresp_txt(self.reference_corresp_path)
        reference_picks = {
            int(idx): (
                np.asarray([float(u), float(v)], dtype=np.float64),
                np.asarray(xyz, dtype=np.float64),
            )
            for idx, (u, v, xyz) in enumerate(zip(corr_i.tolist(), corr_j.tolist(), corr_xyz.tolist()))
        }
        picked = [
            p for p in self.get_picks()["picks"]
            if p.get("las_xyz") is not None and int(p["index"]) in reference_picks
        ]
        if not picked:
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
            )
        picked.sort(key=lambda p: p["index"])
        uv = np.asarray([reference_picks[int(p["index"])][0] for p in picked], dtype=np.float64)
        ref_xyz = np.asarray([reference_picks[int(p["index"])][1] for p in picked], dtype=np.float64)
        las_xyz = np.asarray([p["las_xyz"] for p in picked], dtype=np.float64)
        return uv, ref_xyz, las_xyz

    def _compute_generated_cyl_fit(self, init_cyl_path: str, opt_mode: str = "extr") -> dict[str, Any]:
        init_cyl = Path(init_cyl_path).expanduser()
        if not init_cyl.exists():
            raise ValueError(f"Initial .cyl not found: {init_cyl}")

        using_reference_targets = bool(self.force_generated_cyl_mode and self.use_reference_targets_in_generated_mode and self.reference_corresp_path)
        if using_reference_targets:
            uv_gt, ref_xyz, manual_las = self._picked_reference_target_arrays()
            picked_count = int(manual_las.shape[0])
        else:
            uv_gt, manual_las = self._picked_generated_correspondence_arrays()
            ref_xyz = None
            picked_count = int(manual_las.shape[0])

        if picked_count < MIN_FIT_POINTS:
            status = {
                "state": "insufficient_points",
                "ready": False,
                "mode": "generated_cyl",
                "picked_generated_points": picked_count,
                "min_fit_points": MIN_FIT_POINTS,
                "can_run_fit_feedback": False,
                "can_fit_generated_cyl": True,
                "overlay_preview": None,
                "reprojection_preview": None,
                "cyl_verification_overlay": None,
                "fitted_cyl": None,
                "init_cyl": str(init_cyl),
                "fit_reference_uv": [],
                "fit_projected_uv": [],
            }
            self._write_json(self.fit_json_path, status)
            return status

        gray = load_gray_preview(self.hdr_path).astype(np.float64) / 255.0
        H, W = gray.shape
        cam_init = read_cam(str(init_cyl))

        if using_reference_targets:
            R_las_to_ref, t_ref = fit_rigid_transform(manual_las, ref_xyz)
            corr_xyz = (R_las_to_ref @ manual_las.T).T + t_ref.reshape(1, 3)
            corr_i = uv_gt[:, 0]
            corr_j = uv_gt[:, 1]
        else:
            # Ensure artifact stays in sync with current picks.
            self._write_generated_corresp_txt()
            corr_i, corr_j, corr_xyz = read_corresp(str(self.generated_corresp_path))
            R_las_to_ref = None
            t_ref = None

        cam_opt, stats = calibrate_single(
            corr_i,
            corr_j,
            corr_xyz,
            cam_init,
            opt_mode=opt_mode,
            image_width=W,
            image_height=H,
            max_iters=400,
        )
        write_cyl(cam_opt, str(self.fitted_cyl_path))

        uv_fit = project_vect_safe(corr_xyz, cam_opt)
        residual = uv_fit - np.column_stack((corr_i, corr_j))
        fit_metrics = metrics_from_residual(residual)
        save_manual_projection_plot(
            gray,
            np.column_stack((corr_i, corr_j)),
            uv_fit,
            self.reprojection_preview_path,
            f"{self.path_name.split('_DistStA')[0]} Step{self.step}",
        )

        las = laspy.read(self.projection_las)
        xyz_las = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
        xyz_project = xyz_las
        if using_reference_targets:
            xyz_project = (R_las_to_ref @ xyz_las.T).T + t_ref.reshape(1, 3)
        Pc = (cam_opt.Rot @ xyz_project.T).T + cam_opt.t.reshape(1, 3)
        d = depth_range(Pc)
        ij = project_vect_safe(xyz_project, cam_opt)
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
            "picked_generated_points": picked_count,
            "min_fit_points": MIN_FIT_POINTS,
            "fit_rmse_total": fit_metrics["rmse_total"],
            "fit_rmse_u": fit_metrics["rmse_u"],
            "fit_rmse_v": fit_metrics["rmse_v"],
            "fit_mean_abs_du": fit_metrics["mean_abs_du"],
            "fit_mean_abs_dv": fit_metrics["mean_abs_dv"],
            "optimizer_stats": stats,
            "overlay_preview": str(self.overlay_preview_path),
            "reprojection_preview": str(self.reprojection_preview_path),
            "cyl_verification_overlay": None,
            "fitted_cyl": str(self.fitted_cyl_path),
            "init_cyl": str(init_cyl),
            "fit_reference_uv": np.column_stack((corr_i, corr_j)).tolist(),
            "fit_projected_uv": uv_fit.tolist(),
            "used_reference_targets": using_reference_targets,
            "las_to_reference_rotation": None if R_las_to_ref is None else R_las_to_ref.tolist(),
            "las_to_reference_translation": None if t_ref is None else t_ref.tolist(),
        }
        self._write_json(self.fit_json_path, status)
        return status

    def compute_fit(self, init_cyl_path: str | None = None, opt_mode: str = "extr") -> dict[str, Any]:
        if self.cyl_path and self.corresp_path:
            return self._compute_existing_cyl_fit()
        if not init_cyl_path:
            raise ValueError("Fitting a new .cyl requires an initial .cyl path.")
        return self._compute_generated_cyl_fit(init_cyl_path=init_cyl_path, opt_mode=opt_mode)

    def compute_overlay(self) -> dict[str, Any]:
        raise NotImplementedError("Overlay feedback is reserved for a later phase.")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text())

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
