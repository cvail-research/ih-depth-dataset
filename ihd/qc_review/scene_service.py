import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import spectral as spy


REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = REPO_ROOT / "analysis"
DEFAULT_RESULTS_ROOT = ANALYSIS_ROOT / "lidar_labeling"
QC_ROOT = ANALYSIS_ROOT / "qc_review"
GOOD_VERDICT = "good"
CAUTION_VERDICT = "usable with caution"
BAD_VERDICT = "bad"
VALID_VERDICTS = {GOOD_VERDICT, CAUTION_VERDICT, BAD_VERDICT}
SHARED_REFERENCE_FILENAMES = (
    "ihdepth_qc_reference.png",
    "qc_reference.png",
    "reference.png",
)
SHARED_OVERLAY_FILENAMES = (
    "ihdepth_qc_overlay.png",
    "qc_overlay.png",
    "overlay.png",
)


def path_name_from_key(path_key: str) -> str:
    if path_key.lower().startswith("path"):
        return f"Path{path_key[4:]}_DistStA"
    return path_key


def step_name_from_dir(step_dir: str) -> str:
    parts = step_dir.split("_step")
    if len(parts) == 2:
        return f"Step{int(parts[1])}"
    return step_dir


def resolve_scene_dir(collection: str, path_key: str, step_dir: str, data_root: Path) -> Path | None:
    path_name = path_name_from_key(path_key)
    prefix = path_name.replace("_DistStA", "")
    step = step_name_from_dir(step_dir).replace("Step", "")
    candidates = (
        data_root / collection / path_name / f"{prefix}_Step{step}_DistStA",
        data_root / collection / path_name / f"{prefix}_Step{step}",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def find_first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_hsi_hdr(scene_dir: Path, collection: str, path_key: str, step_dir: str) -> Path | None:
    path_name = path_name_from_key(path_key)
    prefix = path_name.replace("_DistStA", "")
    step = step_name_from_dir(step_dir).replace("Step", "")
    stem = f"{collection.replace('_DistStA', '')}_{prefix}_Step{step}"
    return find_first_existing(
        [
            scene_dir / f"{stem}_LWHSI1_collect0_DistStA.hdr",
            scene_dir / f"{stem}_LWHSI1_DistStA.hdr",
        ]
    )


def resolve_shared_qc_images(scene_dir: Path | None) -> tuple[Path | None, Path | None]:
    if scene_dir is None:
        return None, None
    dataset_references = sorted(scene_dir.glob("*_PseudoBB*_DistStA.png"))
    dataset_overlays = sorted(scene_dir.glob("*_DepthOverlay*_DistStA.png"))
    reference = dataset_references[0] if dataset_references else None
    overlay = dataset_overlays[0] if dataset_overlays else None
    if reference is None:
        reference = find_first_existing([scene_dir / name for name in SHARED_REFERENCE_FILENAMES])
    if overlay is None:
        overlay = find_first_existing([scene_dir / name for name in SHARED_OVERLAY_FILENAMES])
    return reference, overlay


def build_reference_preview(hdr_path: Path, out_path: Path) -> None:
    bsq_path = hdr_path.with_suffix(".bsq")
    img = spy.envi.open(str(hdr_path), str(bsq_path))
    cube = img.load()
    gray = cube.sum(axis=-1).astype(np.float64)
    gray -= gray.min()
    if gray.max() > 0:
        gray /= gray.max()
    gray8 = np.clip(np.round(gray * 255.0), 0, 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(out_path), gray8)
    if not success:
        raise RuntimeError(f"Failed to write reference preview to {out_path}")


@dataclass
class SceneRecord:
    index: int
    scene_label: str
    collection: str
    path_key: str
    step_dir: str
    summary_path: Path | None
    overlay_path: Path
    reference_hdr_path: Path | None
    reference_png_path: Path | None
    summary: dict[str, Any]
    source_kind: str


def scene_uid(collection: str, path_key: str, step_dir: str) -> str:
    return f"{collection}/{path_key}/{step_dir}"


def discover_qc_scenes(results_root: Path, data_root: Path, cache_root: Path) -> list["SceneRecord"]:
    analysis_root = results_root.parent
    annotation_roots = (
        analysis_root / "annotation_workspace",
        analysis_root / "annotation_workspace_nocyl",
    )

    def build_cached_reference_path(collection: str, path_key: str, step_dir: str) -> Path:
        return cache_root / collection / path_key / step_dir / "reference.png"

    def discover_lidar_labeling_scenes() -> dict[str, SceneRecord]:
        scenes: dict[str, SceneRecord] = {}
        summaries = sorted(results_root.rglob("summary.json"))
        for summary_path in summaries:
            rel = summary_path.relative_to(results_root)
            if len(rel.parts) < 4:
                continue
            collection, path_key, step_dir = rel.parts[:3]
            overlay_path = summary_path.parent / "fitted_rigid_overlay.png"
            if not overlay_path.exists():
                continue
            with summary_path.open("r") as f:
                summary = json.load(f)
            scene_dir = resolve_scene_dir(collection, path_key, step_dir, data_root)
            shared_reference_png, shared_overlay_png = resolve_shared_qc_images(scene_dir)
            reference_hdr_path = None
            reference_png_path = None
            chosen_overlay_path = overlay_path
            if scene_dir is not None:
                reference_hdr_path = resolve_hsi_hdr(scene_dir, collection, path_key, step_dir)
                if shared_reference_png is not None:
                    reference_png_path = shared_reference_png
                elif reference_hdr_path is not None:
                    reference_png_path = build_cached_reference_path(collection, path_key, step_dir)
                if shared_overlay_png is not None:
                    chosen_overlay_path = shared_overlay_png
            uid = scene_uid(collection, path_key, step_dir)
            scenes[uid] = SceneRecord(
                index=-1,
                scene_label=str(summary.get("scene_label") or step_dir),
                collection=collection,
                path_key=path_key,
                step_dir=step_dir,
                summary_path=summary_path,
                overlay_path=chosen_overlay_path,
                reference_hdr_path=reference_hdr_path,
                reference_png_path=reference_png_path,
                summary=summary,
                source_kind="lidar_labeling",
            )
        return scenes

    def discover_annotation_workspace_scenes() -> dict[str, SceneRecord]:
        scenes: dict[str, SceneRecord] = {}
        for annotation_root in annotation_roots:
            if not annotation_root.exists():
                continue
            fit_paths = sorted(annotation_root.rglob("fit.json"))
            for fit_path in fit_paths:
                workspace_dir = fit_path.parent
                rel = fit_path.relative_to(annotation_root)
                if len(rel.parts) < 4:
                    continue
                collection, path_key, step_dir = rel.parts[:3]
                if "__" in step_dir:
                    continue
                overlay_path = workspace_dir / "overlay_preview.png"
                if not overlay_path.exists():
                    continue
                scene_json_path = workspace_dir / "scene.json"
                scene_data = {}
                if scene_json_path.exists():
                    with scene_json_path.open("r") as f:
                        scene_data = json.load(f)
                with fit_path.open("r") as f:
                    fit_data = json.load(f)
                if not bool(fit_data.get("ready")):
                    continue

                scene_dir = resolve_scene_dir(collection, path_key, step_dir, data_root)
                shared_reference_png, shared_overlay_png = resolve_shared_qc_images(scene_dir)
                reference_png_path = shared_reference_png
                if reference_png_path is None:
                    reference_png_path = workspace_dir / "image_preview.png"
                    if not reference_png_path.exists():
                        reference_png_path = None
                reference_hdr_path = None
                if scene_data.get("source_paths", {}).get("hsi_hdr"):
                    reference_hdr_path = Path(scene_data["source_paths"]["hsi_hdr"])
                else:
                    if scene_dir is not None:
                        reference_hdr_path = resolve_hsi_hdr(scene_dir, collection, path_key, step_dir)
                chosen_overlay_path = shared_overlay_png or overlay_path
                uid = scene_uid(collection, path_key, step_dir)
                scenes[uid] = SceneRecord(
                    index=-1,
                    scene_label=str(scene_data.get("scene_key") or step_dir),
                    collection=collection,
                    path_key=path_key,
                    step_dir=step_dir,
                    summary_path=fit_path,
                    overlay_path=chosen_overlay_path,
                    reference_hdr_path=reference_hdr_path,
                    reference_png_path=reference_png_path,
                    summary=fit_data,
                    source_kind=annotation_root.name,
                )
        return scenes

    merged = discover_lidar_labeling_scenes()
    annotation = discover_annotation_workspace_scenes()

    for uid, ann_scene in annotation.items():
        if uid in merged:
            existing = merged[uid]
            if existing.reference_png_path is None and ann_scene.reference_png_path is not None:
                existing.reference_png_path = ann_scene.reference_png_path
            if existing.reference_hdr_path is None and ann_scene.reference_hdr_path is not None:
                existing.reference_hdr_path = ann_scene.reference_hdr_path
            continue
        merged[uid] = ann_scene

    scenes = sorted(
        merged.values(),
        key=lambda s: (s.collection, s.path_key, s.step_dir),
    )
    for idx, scene in enumerate(scenes):
        scene.index = idx
    return scenes


class QCReviewSession:
    def __init__(self, reviewer_id: str, scenes: list[SceneRecord], session_dir: Path):
        if not scenes:
            raise ValueError("No scenes available for QC review.")
        self.reviewer_id = reviewer_id
        self.scenes = scenes
        self.session_dir = session_dir
        self.session_path = session_dir / "session.json"
        self.reviews_csv_path = session_dir / "reviews.csv"
        self.current_index = 0
        self.current_started_at = time.monotonic()
        self.reviews = self._load_or_initialize_reviews()
        self._persist()

    def _default_review(self, scene: SceneRecord) -> dict[str, Any]:
        return {
            "scene_label": scene.scene_label,
            "collection": scene.collection,
            "path_key": scene.path_key,
            "step_dir": scene.step_dir,
            "verdict": None,
            "need_more_time": False,
            "total_view_seconds": 0.0,
            "visit_count": 0,
            "last_elapsed_seconds": 0.0,
            "completed": False,
            "updated_at": None,
        }

    def _load_or_initialize_reviews(self) -> dict[str, dict[str, Any]]:
        if self.session_path.exists():
            with self.session_path.open("r") as f:
                data = json.load(f)
            reviews = data.get("reviews", {})
            current_index = int(data.get("current_index", 0))
            if 0 <= current_index < len(self.scenes):
                self.current_index = current_index
            merged = {}
            for scene in self.scenes:
                merged[scene.scene_label] = self._default_review(scene)
                merged[scene.scene_label].update(reviews.get(scene.scene_label, {}))
            return merged
        return {scene.scene_label: self._default_review(scene) for scene in self.scenes}

    def _current_scene(self) -> SceneRecord:
        return self.scenes[self.current_index]

    def _accumulate_current_time(self) -> float:
        elapsed = max(time.monotonic() - self.current_started_at, 0.0)
        review = self.reviews[self._current_scene().scene_label]
        review["total_view_seconds"] = float(review.get("total_view_seconds", 0.0)) + elapsed
        review["last_elapsed_seconds"] = elapsed
        review["visit_count"] = int(review.get("visit_count", 0)) + 1
        return elapsed

    def _persist(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "reviewer_id": self.reviewer_id,
            "current_index": self.current_index,
            "scene_count": len(self.scenes),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reviews": self.reviews,
        }
        with self.session_path.open("w") as f:
            json.dump(data, f, indent=2, sort_keys=True)

        lines = [
            "scene_label,collection,path_key,step_dir,verdict,need_more_time,total_view_seconds,visit_count,last_elapsed_seconds,completed,updated_at"
        ]
        for scene in self.scenes:
            review = self.reviews[scene.scene_label]
            row = [
                scene.scene_label,
                scene.collection,
                scene.path_key,
                scene.step_dir,
                review.get("verdict") or "",
                "1" if review.get("need_more_time") else "0",
                f"{float(review.get('total_view_seconds', 0.0)):.3f}",
                str(int(review.get("visit_count", 0))),
                f"{float(review.get('last_elapsed_seconds', 0.0)):.3f}",
                "1" if review.get("completed") else "0",
                review.get("updated_at") or "",
            ]
            lines.append(",".join(row))
        self.reviews_csv_path.write_text("\n".join(lines) + "\n")

    def _review_summary(self) -> dict[str, Any]:
        completed = sum(1 for r in self.reviews.values() if r.get("completed"))
        remaining = len(self.scenes) - completed
        total_completed_seconds = sum(
            float(r.get("total_view_seconds", 0.0))
            for r in self.reviews.values()
            if r.get("completed")
        )
        average_completed_seconds = (
            total_completed_seconds / completed if completed else None
        )
        estimated_remaining_seconds = (
            average_completed_seconds * remaining if average_completed_seconds is not None else None
        )
        return {
            "scene_count": len(self.scenes),
            "completed_count": completed,
            "remaining_count": remaining,
            "average_completed_seconds": average_completed_seconds,
            "estimated_remaining_seconds": estimated_remaining_seconds,
        }

    def get_scene_payload(self) -> dict[str, Any]:
        scene = self._current_scene()
        review = self.reviews[scene.scene_label]
        return {
            "reviewer_id": self.reviewer_id,
            "current_index": self.current_index,
            "scene": {
                "index": scene.index,
                "scene_label": scene.scene_label,
                "collection": scene.collection,
                "path_key": scene.path_key,
                "step_dir": scene.step_dir,
                "overlay_url": f"/api/scene/{scene.index}/overlay",
                "reference_url": f"/api/scene/{scene.index}/reference",
                "fit_rmse_total": scene.summary.get("fit_rmse_total"),
                "cyl_verify_rmse_total": scene.summary.get("cyl_verify_rmse_total"),
                "num_txt_points": scene.summary.get("num_txt_points"),
            },
            "review": review,
            "progress": self._review_summary(),
        }

    def set_verdict(self, verdict: str, need_more_time: bool | None = None) -> dict[str, Any]:
        if verdict not in VALID_VERDICTS:
            raise ValueError(f"Unsupported verdict: {verdict}")
        review = self.reviews[self._current_scene().scene_label]
        review["verdict"] = verdict
        review["completed"] = True
        review["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._persist()
        return self.get_scene_payload()

    def navigate(self, direction: str) -> dict[str, Any]:
        if direction not in {"next", "prev"}:
            raise ValueError(f"Unsupported navigation direction: {direction}")
        self._accumulate_current_time()
        if direction == "next":
            self.current_index = min(self.current_index + 1, len(self.scenes) - 1)
        else:
            self.current_index = max(self.current_index - 1, 0)
        self.current_started_at = time.monotonic()
        self._persist()
        return self.get_scene_payload()

    def reset_scene_timer(self) -> dict[str, Any]:
        self._accumulate_current_time()
        self.current_started_at = time.monotonic()
        self._persist()
        return self.get_scene_payload()


class QCSceneService:
    def __init__(
        self,
        reviewer_id: str,
        results_root: Path = DEFAULT_RESULTS_ROOT,
        data_root: Path = Path("/disk"),
    ):
        self.reviewer_id = reviewer_id
        self.results_root = results_root
        self.analysis_root = self.results_root.parent
        self.data_root = data_root
        self.cache_root = QC_ROOT / "cache"
        self.session_dir = QC_ROOT / "sessions" / reviewer_id
        self.scenes = discover_qc_scenes(
            results_root=self.results_root,
            data_root=self.data_root,
            cache_root=self.cache_root,
        )
        self.session = QCReviewSession(
            reviewer_id=reviewer_id,
            scenes=self.scenes,
            session_dir=self.session_dir,
        )

    def get_session_state(self) -> dict[str, Any]:
        return self.session.get_scene_payload()

    def get_scene(self, index: int) -> SceneRecord:
        if not (0 <= index < len(self.scenes)):
            raise IndexError(f"Invalid scene index: {index}")
        return self.scenes[index]

    def ensure_reference_preview(self, index: int) -> Path:
        scene = self.get_scene(index)
        if scene.reference_png_path is not None and scene.reference_png_path.exists():
            return scene.reference_png_path
        if scene.reference_png_path is None or scene.reference_hdr_path is None:
            raise FileNotFoundError(f"Missing LWIR reference preview inputs for scene {scene.scene_label}")
        if not scene.reference_png_path.exists():
            build_reference_preview(scene.reference_hdr_path, scene.reference_png_path)
        return scene.reference_png_path

    def get_overlay_path(self, index: int) -> Path:
        scene = self.get_scene(index)
        return scene.overlay_path

    def set_verdict(self, verdict: str) -> dict[str, Any]:
        return self.session.set_verdict(verdict)

    def navigate(self, direction: str) -> dict[str, Any]:
        return self.session.navigate(direction)

    def reset_scene_timer(self) -> dict[str, Any]:
        return self.session.reset_scene_timer()
