import argparse
import csv
from pathlib import Path
from typing import Any

import cv2
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse


REPO_ROOT = Path(__file__).resolve().parents[2]
FAILURE_CACHE_ROOT = REPO_ROOT / "analysis" / "qc_review" / "cache" / "failure_inspection"


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>IH-Depth Failure Inspection</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {
      --bg: #101415;
      --panel: #182124;
      --line: #2d3a3f;
      --text: #eef4f2;
      --muted: #9eb0ad;
      --green: #66c48e;
      --red: #e16055;
      --gold: #e4ac45;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at 12% 0%, rgba(102, 196, 142, 0.10), transparent 28%),
        radial-gradient(circle at 88% 0%, rgba(225, 96, 85, 0.10), transparent 26%),
        var(--bg);
      color: var(--text);
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      overflow: hidden;
    }
    .page {
      height: 100vh;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      gap: 10px;
      padding: 12px;
    }
    .header, .panel, .footer {
      background: rgba(24, 33, 36, 0.96);
      border: 1px solid var(--line);
      border-radius: 14px;
    }
    .header {
      padding: 10px 14px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: center;
    }
    .title {
      font-size: 17px;
      font-weight: 800;
    }
    .meta {
      color: var(--muted);
      font-size: 12px;
      margin-top: 4px;
      line-height: 1.35;
    }
    .badges {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
      align-items: center;
    }
    .badge {
      border-radius: 999px;
      padding: 7px 10px;
      background: #263237;
      color: var(--text);
      font-size: 12px;
      font-weight: 700;
    }
    .badge.red { background: rgba(225, 96, 85, 0.20); color: #ffb8b2; }
    .badge.green { background: rgba(102, 196, 142, 0.18); color: #b5f2ce; }
    .badge.gold { background: rgba(228, 172, 69, 0.18); color: #f4d38e; }
    .content {
      display: grid;
      grid-template-rows: minmax(0, 1fr) minmax(0, 1fr) minmax(104px, 148px);
      gap: 10px;
      min-height: 0;
      overflow: hidden;
      width: 100%;
    }
    .panel {
      min-height: 0;
      padding: 9px;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    h2 {
      margin: 0 0 7px 0;
      font-size: 12px;
      color: var(--muted);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .image-wrap {
      flex: 1;
      min-height: 0;
      background: #0b0f10;
      border: 1px solid #263237;
      border-radius: 10px;
      overflow: hidden;
      display: flex;
      align-items: flex-start;
      justify-content: center;
    }
    img {
      width: 100%;
      height: auto;
      max-height: 100%;
      object-fit: scale-down;
      display: block;
    }
    .inspection-strip {
      min-height: 0;
      width: 100%;
    }
    .table-wrap {
      overflow: auto;
      border-radius: 10px;
      border: 1px solid #263237;
      min-height: 0;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    th, td {
      padding: 7px 8px;
      border-bottom: 1px solid #263237;
      text-align: right;
      white-space: nowrap;
    }
    th:first-child, td:first-child { text-align: left; }
    th {
      position: sticky;
      top: 0;
      background: #202b30;
      z-index: 1;
    }
    tr.fail5 td { color: #ffb8b2; }
    tr.fail1 td { color: #f4d38e; }
    .footer {
      padding: 10px 14px;
      display: grid;
      grid-template-columns: 1fr auto 1fr;
      gap: 10px;
      align-items: center;
    }
    button {
      border: 0;
      border-radius: 999px;
      padding: 11px 16px;
      cursor: pointer;
      font-weight: 800;
      font-size: 14px;
      background: #344349;
      color: var(--text);
    }
    .nav {
      display: flex;
      gap: 10px;
      justify-content: center;
    }
    .hint {
      color: var(--muted);
      font-size: 12px;
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="header">
      <div>
        <div class="title" id="title">Loading...</div>
        <div class="meta" id="meta"></div>
      </div>
      <div class="badges" id="badges"></div>
    </div>

    <div class="content">
      <div class="panel">
        <h2>Depth Overlay</h2>
        <div class="image-wrap"><img id="overlayImg"></div>
      </div>
      <div class="panel">
        <h2>Reference / Correspondence Points</h2>
        <div class="image-wrap"><img id="annotatedReferenceImg"></div>
      </div>
      <div class="inspection-strip">
        <div class="panel">
          <h2>Sampled Correspondences</h2>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Point</th>
                  <th>Picked 3D range m</th>
                  <th>Sampled label range m</th>
                  <th>Abs err m</th>
                  <th>Err %</th>
                  <th>Bin</th>
                  <th>Nearest px</th>
                </tr>
              </thead>
              <tbody id="pointRows"></tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <div class="footer">
      <div class="hint" id="statusText"></div>
      <div class="nav">
        <button id="prevBtn">Back</button>
        <button id="nextBtn">Next</button>
      </div>
      <div class="hint" style="text-align:right;">Use left/right arrows. Rows in red exceed 5%; gold exceeds 1%.</div>
    </div>
  </div>

  <script>
    let state = null;

    async function fetchJson(url, options) {
      const response = await fetch(url, options);
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    }

    function fmt(value, digits=3) {
      if (value === null || value === undefined || value === '' || Number.isNaN(Number(value))) return '';
      return Number(value).toFixed(digits);
    }

    function badge(text, cls='') {
      return `<span class="badge ${cls}">${text}</span>`;
    }

    function render() {
      const scene = state.scene;
      document.getElementById('title').textContent = scene.scene;
      document.getElementById('meta').textContent =
        `${scene.index + 1} of ${state.scene_count} | ${scene.source} | ${scene.fit_path}`;
      document.getElementById('badges').innerHTML = [
        badge(`RMSE ${fmt(scene.fit_rmse_total_px, 2)} px`, scene.rmse_pass_le_10px ? 'green' : 'red'),
        badge(`max ${fmt(scene.distance_max_percent, 2)}%`, scene.distance_pass_5pct ? 'green' : 'red'),
        badge(`${scene.distance_points_gt_5pct} pts >5%`, scene.distance_points_gt_5pct > 0 ? 'red' : 'green'),
        badge(`${scene.distance_points_gt_1pct} pts >1%`, scene.distance_points_gt_1pct > 0 ? 'gold' : 'green')
      ].join('');
      document.getElementById('overlayImg').src = scene.overlay_url;
      document.getElementById('annotatedReferenceImg').src = scene.annotated_reference_url;
      document.getElementById('statusText').textContent =
        `Failure table: ${state.failure_csv} | Showing scenes failing ${state.threshold_percent}% rule`;

      const rows = scene.points.map((p) => {
        let cls = '';
        if (Number(p.error_percent) > 5) cls = 'fail5';
        else if (Number(p.error_percent) > 1) cls = 'fail1';
        return `<tr class="${cls}">
          <td>${p.point_index}</td>
          <td>${fmt(p.picked_range_m)}</td>
          <td>${fmt(p.sampled_overlay_depth_m)}</td>
          <td>${fmt(p.absolute_depth_error_m)}</td>
          <td>${fmt(p.error_percent, 2)}</td>
          <td>${p.distance_bin || ''}</td>
          <td>${fmt(p.nearest_depth_pixel_distance_px, 2)}</td>
        </tr>`;
      }).join('');
      document.getElementById('pointRows').innerHTML = rows || '<tr><td colspan="7">No sampled points.</td></tr>';
    }

    async function loadState() {
      state = await fetchJson('/api/state');
      render();
    }

    async function navigate(direction) {
      state = await fetchJson('/api/navigate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({direction})
      });
      render();
    }

    document.getElementById('nextBtn').onclick = () => navigate('next');
    document.getElementById('prevBtn').onclick = () => navigate('prev');
    window.addEventListener('keydown', (event) => {
      if (event.key === 'ArrowRight') navigate('next');
      if (event.key === 'ArrowLeft') navigate('prev');
    });
    loadState();
  </script>
</body>
</html>
"""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


class FailureInspectionService:
    def __init__(self, failure_csv: Path, point_csv: Path, threshold_percent: float):
        self.failure_csv = failure_csv
        self.point_csv = point_csv
        self.threshold_percent = threshold_percent
        self.current_index = 0
        self.scenes = read_csv(failure_csv)
        self.point_rows = read_csv(point_csv)
        self.points_by_scene: dict[tuple[str, str, str], list[dict[str, str]]] = {}
        for row in self.point_rows:
            key = (row["collection"], row["path"], row["step"])
            self.points_by_scene.setdefault(key, []).append(row)
        if not self.scenes:
            raise ValueError(f"No scenes found in {failure_csv}")

    def _current_scene(self) -> dict[str, str]:
        return self.scenes[self.current_index]

    def _path_for_current(self, kind: str) -> Path:
        scene = self._current_scene()
        if kind == "overlay":
            return Path(scene["disk_overlay"])
        if kind == "reference":
            return Path(scene["disk_reference"])
        if kind == "annotated-reference":
            return self._annotated_reference_path(self.current_index)
        if kind == "reprojection":
            fit_path = Path(scene["fit_path"])
            return fit_path.parent / "reprojection_preview.png"
        raise ValueError(f"Unsupported image kind: {kind}")

    def _reference_base_path(self, scene: dict[str, str]) -> Path:
        fit_path = Path(scene["fit_path"])
        preview = fit_path.parent / "image_preview.png"
        if preview.exists():
            return preview
        return Path(scene["disk_reference"])

    def _points_for_scene(self, scene: dict[str, str]) -> list[dict[str, Any]]:
        key = (scene["collection"], scene["path"], scene["step"])
        points = []
        for row in self.points_by_scene.get(key, []):
            if row.get("status") != "sampled":
                continue
            points.append(
                {
                    "point_index": to_int(row.get("point_index", "")),
                    "picked_u": to_float(row.get("picked_u", "")),
                    "picked_v": to_float(row.get("picked_v", "")),
                    "picked_range_m": to_float(row.get("picked_range_m", "")),
                    "sampled_overlay_depth_m": to_float(row.get("sampled_overlay_depth_m", "")),
                    "absolute_depth_error_m": to_float(row.get("absolute_depth_error_m", "")),
                    "error_percent": to_float(row.get("absolute_depth_error_percent_of_range", "")),
                    "distance_bin": row.get("distance_bin", ""),
                    "nearest_depth_pixel_distance_px": to_float(row.get("nearest_depth_pixel_distance_px", "")),
                }
            )
        points.sort(key=lambda row: (-(row["error_percent"] or -1), row["point_index"]))
        return points

    def _annotated_reference_path(self, index: int) -> Path:
        scene = self.scenes[index]
        out_path = FAILURE_CACHE_ROOT / scene["collection"] / scene["path"] / scene["step"] / "annotated_reference.png"
        source = self._reference_base_path(scene)
        if out_path.exists() and out_path.stat().st_mtime >= source.stat().st_mtime:
            return out_path

        img = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(source)

        height, width = img.shape[:2]
        points = self._points_for_scene(scene)
        for point in points:
            u = point.get("picked_u")
            v = point.get("picked_v")
            if u is None or v is None:
                continue
            x = int(round(float(u)))
            y = int(round(float(v)))
            if not (0 <= x < width and 0 <= y < height):
                continue
            error = point.get("error_percent") or 0.0
            if error > 5:
                color = (80, 80, 255)
            elif error > 1:
                color = (45, 190, 245)
            else:
                color = (130, 220, 120)
            label = str(point["point_index"])
            cv2.circle(img, (x, y), 8, color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(img, (x, y), 9, (10, 10, 10), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(
                img,
                label,
                (x + 11, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_path), img):
            raise RuntimeError(f"Failed to write annotated reference: {out_path}")
        return out_path

    def image_path(self, index: int, kind: str) -> Path:
        if not (0 <= index < len(self.scenes)):
            raise IndexError(f"Invalid scene index: {index}")
        old_index = self.current_index
        self.current_index = index
        try:
            path = self._path_for_current(kind)
        finally:
            self.current_index = old_index
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def state(self) -> dict[str, Any]:
        scene = self._current_scene()
        points = self._points_for_scene(scene)
        return {
            "scene_count": len(self.scenes),
            "failure_csv": str(self.failure_csv),
            "point_csv": str(self.point_csv),
            "threshold_percent": self.threshold_percent,
            "scene": {
                "index": self.current_index,
                "scene": scene.get("scene", ""),
                "source": scene.get("source", ""),
                "fit_path": scene.get("fit_path", ""),
                "fit_rmse_total_px": to_float(scene.get("fit_rmse_total_px", "")),
                "rmse_pass_le_10px": scene.get("rmse_pass_le_10px") == "True",
                "distance_max_percent": to_float(scene.get("distance_max_percent", "")),
                "distance_points_gt_1pct": to_int(scene.get("distance_points_gt_1pct", "")),
                "distance_points_gt_5pct": to_int(scene.get("distance_points_gt_5pct", "")),
                "distance_pass_5pct": scene.get("distance_pass_all_points_le_5pct") == "True",
                "overlay_url": f"/api/scene/{self.current_index}/overlay",
                "reference_url": f"/api/scene/{self.current_index}/reference",
                "reprojection_url": f"/api/scene/{self.current_index}/reprojection",
                "annotated_reference_url": f"/api/scene/{self.current_index}/annotated-reference",
                "points": points,
            },
        }

    def navigate(self, direction: str) -> dict[str, Any]:
        if direction == "next":
            self.current_index = min(self.current_index + 1, len(self.scenes) - 1)
        elif direction == "prev":
            self.current_index = max(self.current_index - 1, 0)
        else:
            raise ValueError(f"Unsupported direction: {direction}")
        return self.state()


def build_app(service: FailureInspectionService) -> FastAPI:
    app = FastAPI(title="IH-Depth Failure Inspection")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(INDEX_HTML)

    @app.get("/api/state")
    async def state() -> JSONResponse:
        return JSONResponse(service.state())

    @app.post("/api/navigate")
    async def navigate(payload: dict[str, str]) -> JSONResponse:
        try:
            return JSONResponse(service.navigate(payload.get("direction", "")))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/scene/{index}/{kind}")
    async def image(index: int, kind: str) -> FileResponse:
        try:
            return FileResponse(service.image_path(index, kind))
        except (IndexError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the IH-Depth failure inspection web app.")
    ap.add_argument(
        "--failure-csv",
        default="analysis/qc_review/reproducible_qc_report/scenes_failing_distance_5pct.csv",
        help="Scene-level failure CSV to inspect.",
    )
    ap.add_argument(
        "--point-csv",
        default="analysis/qc_review/correspondence_distance_errors/per_correspondence_local_depth_errors.csv",
        help="Per-correspondence local depth error CSV.",
    )
    ap.add_argument("--threshold-percent", type=float, default=5.0)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8766)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    service = FailureInspectionService(
        failure_csv=Path(args.failure_csv),
        point_csv=Path(args.point_csv),
        threshold_percent=args.threshold_percent,
    )
    app = build_app(service)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
