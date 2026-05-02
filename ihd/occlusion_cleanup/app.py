import argparse

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from ihd.occlusion_cleanup.scene_service import OcclusionCleanupWorkspace


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>IH-Depth Occlusion Cleanup</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <script type="importmap">
    {
      "imports": {
        "three": "https://unpkg.com/three@0.164.1/build/three.module.js",
        "three/addons/": "https://unpkg.com/three@0.164.1/examples/jsm/"
      }
    }
  </script>
  <style>
    body { font-family: sans-serif; margin: 0; background: #101418; color: #e8eef2; overflow: hidden; }
    .layout { display: grid; grid-template-rows: minmax(0, 1fr) minmax(0, 1fr); gap: 12px; padding: 12px; height: 100vh; box-sizing: border-box; }
    .top-row, .bottom-row { display: grid; gap: 12px; min-height: 0; }
    .top-row { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
    .bottom-row { grid-template-columns: minmax(0, 1fr) minmax(320px, 0.34fr); }
    .panel { background: #182028; border-radius: 10px; padding: 10px; overflow: hidden; display: flex; flex-direction: column; min-height: 0; }
    .panel h2 { margin: 0 0 8px 0; font-size: 16px; }
    .status { font-size: 13px; line-height: 1.4; margin-bottom: 8px; color: #b8c4cc; }
    .visual-stage { flex: 1; min-height: 0; display: flex; align-items: center; justify-content: center; }
    .image-fit { width: 100%; height: auto; max-height: 100%; object-fit: contain; border-radius: 8px; background: #0d1217; display: block; }
    .overlay-stack { display: grid; grid-template-rows: minmax(0, 1fr) minmax(0, 1fr); gap: 10px; min-height: 0; flex: 1; }
    .overlay-block { display: flex; flex-direction: column; min-height: 0; }
    .overlay-label { font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: #9fb0b7; margin: 0 0 6px 0; }
    .controls { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .controls button, .controls input {
      border: 0; border-radius: 6px; padding: 8px 10px; font-size: 14px;
    }
    .controls button { cursor: pointer; background: #34424f; color: #e8eef2; font-weight: 700; }
    .controls button.primary { background: #4fb286; color: #08110d; }
    .controls input { background: #0d1217; color: #e8eef2; border: 1px solid #2b3640; width: 110px; }
    #cloud { width: 100%; height: 100%; min-height: 360px; border-radius: 8px; overflow: hidden; position: relative; }
    .mono { font-family: monospace; font-size: 12px; }
    .note { color: #9fb0b7; font-size: 12px; margin-top: 4px; }
    @media (max-width: 1100px) {
      body { overflow: auto; }
      .layout { height: auto; min-height: 100vh; grid-template-rows: auto auto; }
      .top-row, .bottom-row { grid-template-columns: minmax(0, 1fr); }
    }
  </style>
</head>
<body>
  <div class="layout">
    <div class="top-row">
      <div class="panel">
        <h2>Reference Image</h2>
        <div class="status" id="sceneStatus">Loading scene...</div>
        <div class="visual-stage"><img class="image-fit" id="referenceImage" alt="Reference image"></div>
      </div>
      <div class="panel">
        <h2>Overlay Comparison</h2>
        <div class="status" id="fitStatus">Fit feedback unavailable.</div>
        <div class="overlay-stack">
          <div class="overlay-block">
            <div class="overlay-label">Raw overlay</div>
            <img class="image-fit" id="rawOverlayImage" alt="Raw overlay">
          </div>
          <div class="overlay-block">
            <div class="overlay-label">Cleanup overlay</div>
            <img class="image-fit" id="cleanupOverlayImage" alt="Cleanup overlay" hidden>
          </div>
        </div>
      </div>
    </div>
    <div class="bottom-row">
      <div class="panel">
        <h2>LiDAR</h2>
        <div class="status" id="cloudStatus">Loading point cloud...</div>
        <div id="cloud"></div>
      </div>
      <div class="panel">
        <h2>Cleanup Controls</h2>
        <div class="status" id="pickStatus">Ctrl+click a LiDAR point to center a cleanup box.</div>
        <div class="mono" id="candidatePoint">Selected center: none</div>
        <div class="controls" style="margin-top: 8px;">
          <label class="mono">Half extent (m)</label>
          <input id="halfExtentInput" type="number" min="0.1" step="0.1" value="1.0">
        </div>
        <div class="controls" style="margin-top: 8px;">
          <button class="primary" id="previewCleanupBtn">Preview Cleanup</button>
          <button id="computeFitBtn">Compute Fit</button>
        </div>
        <div class="status" id="cleanupStatus">No cleanup preview yet.</div>
        <div class="note">The raw LiDAR scene is never modified in place. Cleanup outputs are written next to the scene as a derived preview.</div>
      </div>
    </div>
  </div>

  <script type="module">
    import * as THREE from 'three';
    import { TrackballControls } from 'three/addons/controls/TrackballControls.js';

    const fetchJson = async (url, options) => {
      const resp = await fetch(url, options);
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || resp.statusText);
      }
      return resp.json();
    };

    const formatVec = (v) => v.map((x) => Number(x).toFixed(3)).join(', ');

    const sceneStatus = document.getElementById('sceneStatus');
    const fitStatus = document.getElementById('fitStatus');
    const cloudStatus = document.getElementById('cloudStatus');
    const pickStatus = document.getElementById('pickStatus');
    const candidatePointEl = document.getElementById('candidatePoint');
    const cleanupStatus = document.getElementById('cleanupStatus');
    const referenceImage = document.getElementById('referenceImage');
    const rawOverlayImage = document.getElementById('rawOverlayImage');
    const cleanupOverlayImage = document.getElementById('cleanupOverlayImage');
    const halfExtentInput = document.getElementById('halfExtentInput');
    const cloudRoot = document.getElementById('cloud');

    let state = null;
    let pointcloud = null;
    let candidatePoint = null;
    let cloudScene = null;
    let cloudCamera = null;
    let cloudRenderer = null;
    let cloudControls = null;
    let cloudPoints = null;
    let pointPositions = null;
    let pointPickThreshold = 0.12;
    let pointerDownAt = null;

    function buildPointGeometry(xs, ys, zs) {
      const geometry = new THREE.BufferGeometry();
      if (!xs.length) {
        geometry.setAttribute('position', new THREE.Float32BufferAttribute([], 3));
        return geometry;
      }
      const positions = new Float32Array(xs.length * 3);
      for (let i = 0; i < xs.length; i += 1) {
        positions[3 * i] = xs[i];
        positions[3 * i + 1] = ys[i];
        positions[3 * i + 2] = zs[i];
      }
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      return geometry;
    }

    function buildBaseCloud() {
      const geometry = new THREE.BufferGeometry();
      const n = pointcloud.x.length;
      const positions = new Float32Array(n * 3);
      const colors = new Float32Array(n * 3);
      for (let i = 0; i < n; i += 1) {
        positions[3 * i] = pointcloud.x[i];
        positions[3 * i + 1] = pointcloud.y[i];
        positions[3 * i + 2] = pointcloud.z[i];
        const c = pointcloud.intensity_norm[i];
        colors[3 * i] = c;
        colors[3 * i + 1] = c;
        colors[3 * i + 2] = c;
      }
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      geometry.computeBoundingSphere();
      geometry.computeBoundingBox();
      pointPositions = positions;
      const bbox = geometry.boundingBox;
      const size = new THREE.Vector3();
      bbox.getSize(size);
      pointPickThreshold = Math.max(size.length() * 0.0009, 0.02);
      return new THREE.Points(
        geometry,
        new THREE.PointsMaterial({ size: 2.0, sizeAttenuation: false, vertexColors: true, transparent: true, opacity: 0.92 }),
      );
    }

    function onCloudResize() {
      if (!cloudRenderer || !cloudCamera) return;
      const width = cloudRoot.clientWidth;
      const height = Math.max(cloudRoot.clientHeight, 360);
      cloudCamera.aspect = width / height;
      cloudCamera.updateProjectionMatrix();
      cloudRenderer.setSize(width, height, false);
    }

    function animateCloud() {
      requestAnimationFrame(animateCloud);
      if (cloudControls) cloudControls.update();
      if (cloudRenderer && cloudScene && cloudCamera) cloudRenderer.render(cloudScene, cloudCamera);
    }

    function findNearestScreenPoint(event, maxDistancePx) {
      if (!cloudCamera || !pointPositions) return null;
      const rect = cloudRenderer.domElement.getBoundingClientRect();
      const clickX = event.clientX - rect.left;
      const clickY = event.clientY - rect.top;
      const width = rect.width;
      const height = rect.height;
      const projected = new THREE.Vector3();
      let bestIdx = -1;
      let bestDist2 = maxDistancePx * maxDistancePx;
      for (let i = 0; i < pointPositions.length; i += 3) {
        projected.set(pointPositions[i], pointPositions[i + 1], pointPositions[i + 2]).project(cloudCamera);
        if (!Number.isFinite(projected.x) || !Number.isFinite(projected.y) || !Number.isFinite(projected.z)) continue;
        if (projected.z < -1 || projected.z > 1) continue;
        const px = (projected.x * 0.5 + 0.5) * width;
        const py = (-projected.y * 0.5 + 0.5) * height;
        const dx = px - clickX;
        const dy = py - clickY;
        const dist2 = dx * dx + dy * dy;
        if (dist2 < bestDist2) {
          bestDist2 = dist2;
          bestIdx = i / 3;
        }
      }
      if (bestIdx < 0) return null;
      return { point: [pointPositions[3 * bestIdx], pointPositions[3 * bestIdx + 1], pointPositions[3 * bestIdx + 2]] };
    }

    function updateStatus() {
      sceneStatus.textContent = `${state.collection} / ${state.path_name} / Step${state.step}`;
      cloudStatus.textContent = pointcloud
        ? `Cloud: ${pointcloud.display_source} | points rendered: ${pointcloud.point_count} / ${pointcloud.original_point_count}`
        : 'Points rendered: ...';
      pickStatus.textContent = candidatePoint
        ? `Selected center: ${formatVec(candidatePoint)}`
        : 'Ctrl+click a LiDAR point to center a cleanup box.';
      candidatePointEl.textContent = candidatePoint ? `Selected center: ${formatVec(candidatePoint)}` : 'Selected center: none';
      if (state.cleanup_preview) {
        cleanupStatus.textContent =
          `Removed points: ${state.cleanup_preview.removed_points} | ` +
          `Kept points: ${state.cleanup_preview.kept_points} | ` +
          `Half extent: ${Number(state.cleanup_preview.half_extent_m).toFixed(2)} m`;
      } else {
        cleanupStatus.textContent = 'No cleanup preview yet.';
      }
      if (state.fit && state.fit.ready) {
        fitStatus.textContent = `Fit RMSE: ${Number(state.fit.fit_rmse_total).toFixed(3)} px`;
      } else {
        fitStatus.textContent = 'Fit unavailable. Compute fit first if needed.';
      }
      referenceImage.src = `/api/image?ts=${encodeURIComponent(Date.now())}`;
      rawOverlayImage.src = state.source_overlay_url ? `${state.source_overlay_url}?ts=${encodeURIComponent(Date.now())}` : '';
      rawOverlayImage.hidden = !state.source_overlay_url;
      cleanupOverlayImage.hidden = !state.cleanup_preview;
      if (state.cleanup_preview) {
        cleanupOverlayImage.src = `/api/artifacts/cleanup_overlay.png?ts=${encodeURIComponent(state.cleanup_preview.updated_at || Date.now())}`;
      }
    }

    function renderPointCloud() {
      if (cloudRenderer) return;
      cloudScene = new THREE.Scene();
      cloudScene.background = new THREE.Color('#182028');
      cloudCamera = new THREE.PerspectiveCamera(55, 1, 0.01, 5000);
      cloudCamera.up.set(0, 0, 1);
      cloudRenderer = new THREE.WebGLRenderer({antialias: true, powerPreference: 'high-performance'});
      cloudRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
      cloudRoot.innerHTML = '';
      cloudRoot.appendChild(cloudRenderer.domElement);
      cloudControls = new TrackballControls(cloudCamera, cloudRenderer.domElement);
      cloudControls.rotateSpeed = 1.05;
      cloudControls.zoomSpeed = 0.85;
      cloudControls.panSpeed = 0.4;
      cloudControls.staticMoving = true;
      cloudControls.dynamicDampingFactor = 0.12;
      cloudPoints = buildBaseCloud();
      cloudScene.add(cloudPoints);
      cloudScene.add(new THREE.AxesHelper(5));
      const bbox = cloudPoints.geometry.boundingBox.clone();
      const center = new THREE.Vector3();
      const size = new THREE.Vector3();
      bbox.getCenter(center);
      bbox.getSize(size);
      const radius = Math.max(size.length() * 0.6, 10);
      cloudCamera.position.set(center.x + radius, center.y + radius, center.z + radius);
      cloudControls.target.copy(center);
      cloudCamera.lookAt(center);
      cloudControls.update();
      cloudRenderer.domElement.addEventListener('pointerdown', (event) => {
        pointerDownAt = {x: event.clientX, y: event.clientY};
      });
      cloudRenderer.domElement.addEventListener('pointerup', async (event) => {
        if (!pointerDownAt) return;
        const dx = event.clientX - pointerDownAt.x;
        const dy = event.clientY - pointerDownAt.y;
        pointerDownAt = null;
        if ((dx * dx + dy * dy) > 16) return;
        if (!event.ctrlKey) return;
        const hit = findNearestScreenPoint(event, 18);
        if (!hit) return;
        candidatePoint = hit.point;
        updateStatus();
      });
      window.addEventListener('resize', onCloudResize);
      onCloudResize();
      animateCloud();
    }

    async function loadScene() {
      state = await fetchJson('/api/scene');
      referenceImage.onload = () => updateStatus();
      referenceImage.src = `/api/image?ts=${encodeURIComponent(Date.now())}`;
      rawOverlayImage.src = state.source_overlay_url ? `${state.source_overlay_url}?ts=${encodeURIComponent(Date.now())}` : '';
      if (state.cleanup_preview) {
        cleanupOverlayImage.hidden = false;
        cleanupOverlayImage.src = `/api/artifacts/cleanup_overlay.png?ts=${encodeURIComponent(state.cleanup_preview.updated_at || Date.now())}`;
      }
      pointcloud = await fetchJson('/api/pointcloud');
      renderPointCloud();
      updateStatus();
    }

    document.getElementById('computeFitBtn').onclick = async () => {
      state.fit = await fetchJson('/api/fit', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({})});
      state = await fetchJson('/api/scene');
      updateStatus();
    };

    document.getElementById('previewCleanupBtn').onclick = async () => {
      if (!candidatePoint) {
        alert('Pick a LiDAR point first.');
        return;
      }
      const halfExtent = Number(halfExtentInput.value);
      if (!Number.isFinite(halfExtent) || halfExtent <= 0) {
        alert('Enter a positive half extent in meters.');
        return;
      }
      state.cleanup_preview = await fetchJson('/api/cleanup-preview', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({center_xyz: candidatePoint, half_extent_m: halfExtent}),
      });
      updateStatus();
    };

    loadScene().catch((err) => {
      console.error(err);
      sceneStatus.textContent = 'Failed to load scene: ' + err.message;
    });
  </script>
</body>
</html>
"""


class CleanupPreviewPayload(BaseModel):
    center_xyz: list[float]
    half_extent_m: float = 1.0


def create_app(workspace: OcclusionCleanupWorkspace) -> FastAPI:
    workspace.prepare()
    app = FastAPI(title="IH-Depth Occlusion Cleanup")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(INDEX_HTML)

    @app.get("/api/scene")
    async def get_scene() -> JSONResponse:
        payload = workspace.get_scene_payload()
        payload["session"] = workspace.get_session()
        payload["fit"] = workspace.get_fit_status()
        payload["source_overlay_url"] = "/api/source-overlay" if workspace.fit_json_path.exists() else None
        return JSONResponse(payload)

    @app.get("/api/image")
    async def get_image() -> FileResponse:
        return FileResponse(workspace.image_preview_path)

    @app.get("/api/source-overlay")
    async def get_source_overlay() -> FileResponse:
        if not workspace.source.overlay_preview_path.exists():
            raise HTTPException(status_code=404, detail="Source overlay is not available yet.")
        return FileResponse(workspace.source.overlay_preview_path)

    @app.get("/api/pointcloud")
    async def get_pointcloud() -> JSONResponse:
        try:
            return JSONResponse(workspace.get_pointcloud_payload())
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/fit")
    async def get_fit() -> JSONResponse:
        return JSONResponse(workspace.get_fit_status())

    @app.post("/api/fit")
    async def compute_fit() -> JSONResponse:
        return JSONResponse(workspace.compute_fit())

    @app.post("/api/cleanup-preview")
    async def cleanup_preview(payload: CleanupPreviewPayload) -> JSONResponse:
        try:
            return JSONResponse(workspace.preview_cleanup(payload.center_xyz, payload.half_extent_m))
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/artifacts/{name}")
    async def get_artifact(name: str) -> FileResponse:
        allowed = {
            "cleanup_raw_overlay.png": workspace.cleanup_raw_overlay_path,
            "cleanup_overlay.png": workspace.cleanup_overlay_path,
        }
        path = allowed.get(name)
        if path is None or not path.exists():
            raise HTTPException(status_code=404, detail=f"Artifact not found: {name}")
        return FileResponse(path)

    return app


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the IH-Depth occlusion cleanup web app.")
    ap.add_argument("--collection", required=True)
    ap.add_argument("--path-name", required=True)
    ap.add_argument("--step", required=True, type=int)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", default=8000, type=int)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    workspace = OcclusionCleanupWorkspace(args.collection, args.path_name, args.step)
    app = create_app(workspace)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
