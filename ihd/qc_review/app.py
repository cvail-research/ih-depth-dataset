import argparse
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from ihd.qc_review.scene_service import QCSceneService


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>IH-Depth QC Review</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {
      --bg: #0c1214;
      --panel: #162025;
      --line: #29363d;
      --text: #eef4f6;
      --muted: #9fb0b7;
      --good: #74d39f;
      --warn: #f0b43c;
      --bad: #ef6c67;
      --red: #ff4d4d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(116, 211, 159, 0.08), transparent 28%),
        radial-gradient(circle at top right, rgba(240, 180, 60, 0.08), transparent 24%),
        var(--bg);
      color: var(--text);
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      overflow: hidden;
    }
    .page {
      height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 14px;
      padding: 14px;
      overflow: hidden;
    }
    .header, .footer, .panel {
      background: rgba(22, 32, 37, 0.95);
      border: 1px solid var(--line);
      border-radius: 14px;
    }
    .header {
      padding: 10px 14px;
      display: grid;
      grid-template-columns: 1.9fr 1.1fr auto;
      gap: 10px;
      align-items: center;
    }
    .title {
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0.02em;
    }
    .subtitle, .meta {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }
    .timer-box {
      justify-self: end;
      text-align: right;
    }
    .timer-label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .timer {
      font-size: 32px;
      font-weight: 800;
      line-height: 1;
      margin-top: 2px;
    }
    .timer.overdue {
      color: var(--red);
      animation: pulse 1s steps(2, end) infinite;
    }
    @keyframes pulse {
      50% { opacity: 0.55; }
    }
    .content {
      display: grid;
      grid-template-rows: minmax(0, 1fr) minmax(0, 1fr);
      gap: 14px;
      min-height: 0;
      overflow: hidden;
    }
    .panel {
      padding: 10px;
      display: flex;
      flex-direction: column;
      min-height: 0;
      overflow: hidden;
    }
    .panel h2 {
      margin: 0 0 8px 0;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }
    .image-wrap {
      flex: 1;
      min-height: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      border-radius: 10px;
      background: #0a0f12;
      border: 1px solid #1e2a31;
    }
    .image-wrap img {
      width: 100%;
      max-height: 100%;
      height: auto;
      display: block;
    }
    .footer {
      padding: 10px 14px;
      display: grid;
      grid-template-columns: 1fr auto 1fr;
      gap: 10px;
      align-items: center;
    }
    .progress {
      font-size: 14px;
      font-weight: 600;
    }
    .footer-status {
      display: flex;
      align-items: baseline;
      gap: 10px;
      flex-wrap: wrap;
    }
    .footer-note {
      color: var(--muted);
      font-size: 12px;
    }
    .actions, .nav {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }
    .actions { justify-content: center; }
    .nav { justify-content: flex-end; }
    button {
      border: 0;
      border-radius: 999px;
      padding: 12px 16px;
      cursor: pointer;
      font-weight: 700;
      font-size: 14px;
      transition: transform 0.08s ease, opacity 0.08s ease;
    }
    button:active { transform: translateY(1px); }
    .good { background: var(--good); color: #0b1a12; }
    .caution { background: var(--warn); color: #251900; }
    .bad { background: var(--bad); color: #220a09; }
    .secondary { background: #31414a; color: var(--text); }
    .secondary.active {
      outline: 2px solid var(--text);
      background: #41535d;
    }
    @media (max-width: 1100px) {
      .header, .footer {
        grid-template-columns: 1fr;
      }
      .timer-box {
        justify-self: start;
        text-align: left;
      }
      .actions, .nav {
        justify-content: flex-start;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="header">
      <div>
        <div class="title" id="sceneLabel">Loading...</div>
        <div class="subtitle" id="sceneMeta">Preparing scene metadata...</div>
      </div>
      <div class="meta" id="progressMeta">Loading progress...</div>
      <div class="timer-box">
        <div class="timer-label">Scene Timer</div>
        <div class="timer" id="timerText">00:00</div>
      </div>
    </div>

    <div class="content">
      <div class="panel">
        <h2>Projected LiDAR Overlay</h2>
        <div class="image-wrap"><img id="overlayImage" alt="Overlay image"></div>
      </div>
      <div class="panel">
        <h2>Reference Pseudobroadband HSI</h2>
        <div class="image-wrap"><img id="referenceImage" alt="Reference image"></div>
      </div>
    </div>

    <div class="footer">
      <div>
        <div class="footer-status">
          <div class="progress" id="decisionText">No verdict yet.</div>
          <div class="footer-note" id="footerNote"></div>
        </div>
      </div>
      <div class="actions">
        <button class="good" id="goodBtn">1 Good</button>
        <button class="caution" id="cautionBtn">2 Usable with caution</button>
        <button class="bad" id="badBtn">3 Bad</button>
      </div>
      <div class="nav">
        <button class="secondary" id="prevBtn">Back</button>
        <button class="secondary" id="nextBtn">Next</button>
      </div>
    </div>
  </div>

  <script>
    const TIMER_WARNING_SECONDS = 30.0;
    let state = null;
    let timerInterval = null;
    let sceneLoadedAtMs = null;
    let lastSceneIndex = null;

    async function fetchJson(url, options) {
      const response = await fetch(url, options);
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || response.statusText);
      }
      return response.json();
    }

    function formatSeconds(value) {
      const total = Math.max(0, Math.floor(value));
      const mins = Math.floor(total / 60);
      const secs = total % 60;
      return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    }

    function currentElapsedSeconds() {
      if (!state || sceneLoadedAtMs === null) return 0;
      return (state.review.total_view_seconds || 0) + ((Date.now() - sceneLoadedAtMs) / 1000.0);
    }

    function updateTimer() {
      const timerText = document.getElementById('timerText');
      const elapsed = currentElapsedSeconds();
      timerText.textContent = formatSeconds(elapsed);
      timerText.classList.toggle('overdue', elapsed > TIMER_WARNING_SECONDS);
    }

    function render() {
      const scene = state.scene;
      const progress = state.progress;
      const sceneChanged = lastSceneIndex !== scene.index;
      document.getElementById('sceneLabel').textContent =
        `${scene.collection} / ${scene.path_key} / ${scene.step_dir}`;
      document.getElementById('sceneMeta').textContent =
        `Reviewer: ${state.reviewer_id} | Scene ${scene.index + 1} of ${progress.scene_count} | Remaining scenes: ${progress.remaining_count}`;
      document.getElementById('progressMeta').textContent =
        '';
      document.getElementById('referenceImage').src = scene.reference_url;
      document.getElementById('overlayImage').src = scene.overlay_url;
      document.getElementById('decisionText').textContent = state.review.verdict
        ? `Verdict: ${state.review.verdict}`
        : 'No verdict yet.';
      document.getElementById('footerNote').textContent =
        'Use keys 1, 2, 3, left arrow, and right arrow.';
      if (sceneChanged || sceneLoadedAtMs === null) {
        sceneLoadedAtMs = Date.now();
      }
      lastSceneIndex = scene.index;
      updateTimer();
      if (timerInterval) clearInterval(timerInterval);
      timerInterval = setInterval(updateTimer, 250);
    }

    async function loadState() {
      state = await fetchJson('/api/state');
      render();
    }

    async function setVerdict(verdict) {
      state = await fetchJson('/api/verdict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          verdict
        })
      });
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

    document.getElementById('goodBtn').onclick = () => setVerdict('good');
    document.getElementById('cautionBtn').onclick = () => setVerdict('usable with caution');
    document.getElementById('badBtn').onclick = () => setVerdict('bad');
    document.getElementById('nextBtn').onclick = () => navigate('next');
    document.getElementById('prevBtn').onclick = () => navigate('prev');

    window.addEventListener('keydown', (event) => {
      if (event.target && ['INPUT', 'TEXTAREA'].includes(event.target.tagName)) return;
      if (event.key === '1') setVerdict('good');
      if (event.key === '2') setVerdict('usable with caution');
      if (event.key === '3') setVerdict('bad');
      if (event.key === 'ArrowRight') navigate('next');
      if (event.key === 'ArrowLeft') navigate('prev');
    });

    loadState();
  </script>
</body>
</html>
"""


class VerdictPayload(BaseModel):
    verdict: str


class NavigationPayload(BaseModel):
    direction: str


def build_app(service: QCSceneService) -> FastAPI:
    app = FastAPI(title="IH-Depth QC Review")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(INDEX_HTML)

    @app.get("/api/state")
    async def state() -> JSONResponse:
        return JSONResponse(service.get_session_state())

    @app.post("/api/verdict")
    async def verdict(payload: VerdictPayload) -> JSONResponse:
        try:
            return JSONResponse(service.set_verdict(payload.verdict))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/navigate")
    async def navigate(payload: NavigationPayload) -> JSONResponse:
        try:
            return JSONResponse(service.navigate(payload.direction))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/reset-timer")
    async def reset_timer() -> JSONResponse:
        return JSONResponse(service.reset_scene_timer())

    @app.get("/api/scene/{index}/overlay")
    async def overlay(index: int) -> FileResponse:
        try:
            return FileResponse(service.get_overlay_path(index))
        except IndexError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/scene/{index}/reference")
    async def reference(index: int) -> FileResponse:
        try:
            return FileResponse(service.ensure_reference_preview(index))
        except (IndexError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the IH-Depth QC review web app.")
    ap.add_argument("--reviewer-id", required=True, help="Reviewer identifier used for saved outputs.")
    ap.add_argument("--results-root", default=str(Path("analysis/lidar_labeling")), help="Primary results root for labeled scenes; annotation workspace pools are merged automatically.")
    ap.add_argument("--data-root", default="/disk", help="Dataset root containing original DARPA scene folders.")
    ap.add_argument("--host", default="0.0.0.0", help="Host to bind the web server to.")
    ap.add_argument("--port", type=int, default=8765, help="Port to bind the web server to.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    service = QCSceneService(
        reviewer_id=args.reviewer_id,
        results_root=Path(args.results_root),
        data_root=Path(args.data_root),
    )
    app = build_app(service)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
