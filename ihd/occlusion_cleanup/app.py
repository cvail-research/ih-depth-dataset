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
    .region-list { display: grid; gap: 8px; margin-top: 10px; max-height: 220px; overflow: auto; }
    .region-row { display: flex; align-items: center; justify-content: space-between; gap: 8px; background: #11171d; border: 1px solid #23303a; border-radius: 8px; padding: 8px 10px; }
    .region-row .meta { font-size: 12px; color: #cbd7de; }
    .region-row button { padding: 6px 8px; font-size: 12px; }
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
          <button class="primary" id="addRegionBtn">Add Region</button>
          <button id="recomputePreviewBtn">Preview All</button>
          <button id="undoRegionBtn">Undo Last</button>
          <button id="computeFitBtn">Recompute Fit</button>
        </div>
        <div class="note">Recompute Fit is only needed if the scene does not have a fit yet or if you changed correspondences. Cleanup itself does not depend on it once a fit exists.</div>
        <div class="region-list" id="regionList"></div>
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
    const regionList = document.getElementById('regionList');
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
    let maxPickScreenDistancePx = 18;
    let pointerDownAt = null;
    let statusFlash = '';
    let candidateSource = 'unknown';
    let overlayPickRadiusPx = 14;

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
      return {
        distancePx: Math.sqrt(bestDist2),
        point: [pointPositions[3 * bestIdx], pointPositions[3 * bestIdx + 1], pointPositions[3 * bestIdx + 2]],
      };
    }

    function findNearestOverlayPoint(event) {
      if (!state || !state.fit || !state.fit.ready || !pointcloud || !pointcloud.projected_u || !pointcloud.projected_u.length) {
        return null;
      }
      const img = event.currentTarget;
      const rect = img.getBoundingClientRect();
      const naturalWidth = state.fit.default_camera?.image_width || img.naturalWidth || rect.width;
      const naturalHeight = state.fit.default_camera?.image_height || img.naturalHeight || rect.height;
      const clickX = (event.clientX - rect.left) * (naturalWidth / rect.width);
      const clickY = (event.clientY - rect.top) * (naturalHeight / rect.height);
      let bestIdx = -1;
      let bestDist2 = overlayPickRadiusPx * overlayPickRadiusPx;
      let bestDepth = -Infinity;
      for (let i = 0; i < pointcloud.projected_u.length; i += 1) {
        if (pointcloud.projected_valid && !pointcloud.projected_valid[i]) continue;
        const u = pointcloud.projected_u[i];
        const v = pointcloud.projected_v[i];
        if (!Number.isFinite(u) || !Number.isFinite(v)) continue;
        const dx = u - clickX;
        const dy = v - clickY;
        const dist2 = dx * dx + dy * dy;
        const depth = Number(pointcloud.projected_depth[i]);
        // Occlusion cleanup should target hidden/background leakage, so we
        // prioritize the farthest projected point inside the click radius.
        if (depth > bestDepth || (Math.abs(depth - bestDepth) <= 1e-6 && dist2 < bestDist2)) {
          bestDepth = depth;
          bestDist2 = dist2;
          bestIdx = i;
        }
      }
      if (bestIdx < 0) return null;
      return {
        distancePx: Math.sqrt(bestDist2),
        point: [pointcloud.x[bestIdx], pointcloud.y[bestIdx], pointcloud.z[bestIdx]],
      };
    }

    function setRotationCenterFromEvent(event) {
      if (!cloudPoints) return;
      const hit = findNearestScreenPoint(event, maxPickScreenDistancePx);
      if (!hit) {
        statusFlash = 'no point found to use as rotation center';
        updateStatus();
        return;
      }
      const center = new THREE.Vector3(hit.point[0], hit.point[1], hit.point[2]);
      const offset = new THREE.Vector3().subVectors(cloudCamera.position, cloudControls.target);
      cloudControls.target.copy(center);
      cloudCamera.position.copy(center.clone().add(offset));
      cloudCamera.updateProjectionMatrix();
      statusFlash = `rotation center moved (${hit.distancePx.toFixed(1)} px)`;
      cloudControls.update();
      updateStatus();
    }

    function updateStatus() {
      sceneStatus.textContent = `${state.collection} / ${state.path_name} / Step${state.step}`;
      cloudStatus.textContent = pointcloud
        ? `Cloud: ${pointcloud.display_source} | points rendered: ${pointcloud.point_count} / ${pointcloud.original_point_count}`
        : 'Points rendered: ...';
      pickStatus.textContent = statusFlash || (candidatePoint
        ? `Selected center: ${formatVec(candidatePoint)}`
        : 'Ctrl+click a LiDAR point to center a cleanup box.');
      candidatePointEl.textContent = candidatePoint ? `Selected center: ${formatVec(candidatePoint)}` : 'Selected center: none';
      if (state.cleanup_preview) {
        const regionCount = state.cleanup_preview.cleanup_region_count ?? state.cleanup_preview.regions?.length ?? 0;
        const modeSummary = state.cleanup_preview.selection_mode_summary
          ? Object.entries(state.cleanup_preview.selection_mode_summary).map(([k, v]) => `${k}:${v}`).join(', ')
          : '';
        cleanupStatus.textContent =
          `Regions: ${regionCount}${modeSummary ? ` | Modes: ${modeSummary}` : ''} | ` +
          `Removed points: ${state.cleanup_preview.removed_points} | ` +
          `Kept points: ${state.cleanup_preview.kept_points} | ` +
          `Half extent: ${Number(state.cleanup_preview.half_extent_m).toFixed(2)} m`;
        const regions = state.cleanup_preview.regions || [];
        regionList.innerHTML = regions.length
          ? regions.map((region, idx) => {
              const center = (region.center_xyz || []).map((v) => Number(v).toFixed(3)).join(', ');
              return `
                <div class="region-row">
                  <div class="meta">#${idx + 1} [${region.region_id || 'legacy'}] | ${region.selection_mode || 'unknown'} | r=${Number(region.half_extent_m).toFixed(2)} m | ${center}</div>
                  <button data-region-delete="${region.region_id || idx}">Delete</button>
                </div>`;
            }).join('')
          : '<div class="note">No cleanup regions yet.</div>';
      } else {
        cleanupStatus.textContent = 'No cleanup preview yet.';
        regionList.innerHTML = '<div class="note">No cleanup regions yet.</div>';
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

    async function setCandidateFromOverlay(event) {
      const hit = findNearestOverlayPoint(event);
      if (!hit) {
        statusFlash = 'no projected point close enough to the overlay click';
        updateStatus();
        return;
      }
      candidatePoint = hit.point;
      candidateSource = 'overlay';
      statusFlash = `overlay point selected (${hit.distancePx.toFixed(1)} px)`;
      updateStatus();
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
        if (event.shiftKey) {
          setRotationCenterFromEvent(event);
          return;
        }
        if (!event.ctrlKey) return;
        const hit = findNearestScreenPoint(event, maxPickScreenDistancePx);
        if (!hit) return;
        candidatePoint = hit.point;
        candidateSource = 'lidar';
        statusFlash = `candidate selected (${hit.distancePx.toFixed(1)} px)`;
        updateStatus();
      });
      rawOverlayImage.addEventListener('click', async (event) => {
        await setCandidateFromOverlay(event);
      });
      cleanupOverlayImage.addEventListener('click', async (event) => {
        await setCandidateFromOverlay(event);
      });
      window.addEventListener('resize', onCloudResize);
      onCloudResize();
      animateCloud();
    }

    function refreshCloudGeometry() {
      if (!cloudScene || !cloudPoints) return;
      const newPoints = buildBaseCloud();
      const oldPoints = cloudPoints;
      cloudScene.remove(oldPoints);
      cloudPoints = newPoints;
      cloudScene.add(cloudPoints);
      if (oldPoints.geometry) oldPoints.geometry.dispose();
    }

    async function refreshPointCloudFromServer() {
      pointcloud = await fetchJson('/api/pointcloud');
      if (!cloudRenderer) {
        renderPointCloud();
      } else {
        refreshCloudGeometry();
      }
    }

    async function loadScene() {
      state = await fetchJson('/api/scene');
      referenceImage.onload = () => updateStatus();
      referenceImage.src = `/api/image?ts=${encodeURIComponent(Date.now())}`;
      rawOverlayImage.src = state.source_overlay_url ? `${state.source_overlay_url}?ts=${encodeURIComponent(Date.now())}` : '';
      if (state.cleanup_preview) {
        cleanupOverlayImage.hidden = false;
        cleanupOverlayImage.src = `/api/artifacts/cleanup_overlay.png?ts=${encodeURIComponent(state.cleanup_preview.updated_at || Date.now())}`;
      } else {
        cleanupOverlayImage.hidden = true;
      }
      await refreshPointCloudFromServer();
      updateStatus();
    }

    document.getElementById('computeFitBtn').onclick = async () => {
      state.fit = await fetchJson('/api/fit', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({})});
      state = await fetchJson('/api/scene');
      await refreshPointCloudFromServer();
      updateStatus();
    };

    document.getElementById('addRegionBtn').onclick = async () => {
      if (!candidatePoint) {
        alert('Pick a LiDAR point first.');
        return;
      }
      const halfExtent = Number(halfExtentInput.value);
      if (!Number.isFinite(halfExtent) || halfExtent <= 0) {
        alert('Enter a positive half extent in meters.');
        return;
      }
      state.cleanup_preview = await fetchJson('/api/cleanup-region-add', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({center_xyz: candidatePoint, half_extent_m: halfExtent, selection_mode: candidateSource}),
      });
      state = await fetchJson('/api/scene');
      await refreshPointCloudFromServer();
      updateStatus();
    };

    document.getElementById('recomputePreviewBtn').onclick = async () => {
      state.cleanup_preview = await fetchJson('/api/cleanup-preview', {method: 'POST'});
      state = await fetchJson('/api/scene');
      await refreshPointCloudFromServer();
      updateStatus();
    };

    document.getElementById('undoRegionBtn').onclick = async () => {
      state.cleanup_preview = await fetchJson('/api/cleanup-region-undo', {method: 'POST'});
      state = await fetchJson('/api/scene');
      await refreshPointCloudFromServer();
      updateStatus();
    };

    regionList.addEventListener('click', async (event) => {
      const button = event.target.closest('button[data-region-delete]');
      if (!button) return;
      const regionId = button.getAttribute('data-region-delete');
      if (!regionId) return;
      state.cleanup_preview = await fetchJson(`/api/cleanup-region-delete/${encodeURIComponent(regionId)}`, {method: 'POST'});
      state = await fetchJson('/api/scene');
      await refreshPointCloudFromServer();
      updateStatus();
    });

    loadScene().catch((err) => {
      console.error(err);
      sceneStatus.textContent = 'Failed to load scene: ' + err.message;
    });
  </script>
</body>
</html>
"""


class CleanupRegionPayload(BaseModel):
    center_xyz: list[float]
    half_extent_m: float = 1.0
    selection_mode: str = "unknown"


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
    async def cleanup_preview() -> JSONResponse:
        try:
            return JSONResponse(workspace.recompute_cleanup_preview())
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/cleanup-region-add")
    async def cleanup_region_add(payload: CleanupRegionPayload) -> JSONResponse:
        try:
            return JSONResponse(
                workspace.add_cleanup_region(payload.center_xyz, payload.half_extent_m, payload.selection_mode)
            )
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/cleanup-region-undo")
    async def cleanup_region_undo() -> JSONResponse:
        try:
            return JSONResponse(workspace.undo_cleanup_region())
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/cleanup-region-delete/{index}")
    async def cleanup_region_delete(index: str) -> JSONResponse:
        try:
            return JSONResponse(workspace.remove_cleanup_region(index))
        except (ValueError, FileNotFoundError, IndexError) as exc:
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
