import argparse
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from ihd.annotation_workspace.scene_service import SceneWorkspace


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Annotation Workspace</title>
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
    body { font-family: sans-serif; margin: 0; background: #101418; color: #e8eef2; }
    .layout { display: grid; grid-template-rows: minmax(300px, 0.9fr) minmax(320px, 1.1fr); gap: 12px; padding: 12px; height: 100vh; box-sizing: border-box; }
    .top-row { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px; min-height: 0; }
    .bottom-row { display: grid; grid-template-columns: minmax(0, 1fr) minmax(180px, 0.28fr); gap: 12px; min-height: 0; }
    .panel { background: #182028; border-radius: 10px; padding: 10px; overflow: hidden; display: flex; flex-direction: column; }
    .panel h2 { margin: 0 0 8px 0; font-size: 16px; }
    .image-panel, .fit-panel, .cloud-panel, .targets-panel { min-height: 0; }
    .visual-stage { flex: 1; min-height: 0; display: flex; align-items: center; justify-content: center; }
    #imageCanvas { width: 100%; height: auto; max-height: 100%; background: #0d1217; cursor: crosshair; border-radius: 8px; display: block; }
    #cloud { width: 100%; height: 100%; min-height: 400px; border-radius: 8px; overflow: hidden; position: relative; }
    .status { font-size: 13px; line-height: 1.4; margin-bottom: 8px; color: #b8c4cc; }
    .controls button { margin: 4px 4px 0 0; padding: 8px 10px; border: 0; border-radius: 6px; cursor: pointer; }
    .controls button.primary { background: #4fb286; color: #08110d; font-weight: 700; }
    .controls button.warn { background: #e2a72e; color: #221500; font-weight: 700; }
    .controls button.danger { background: #e25c5c; color: #200707; font-weight: 700; }
    .controls button.secondary { background: #34424f; color: #e8eef2; }
    #targetList { overflow: auto; border-top: 1px solid #2b3640; margin-top: 10px; padding-top: 8px; }
    .target { padding: 8px; border-radius: 6px; cursor: pointer; margin-bottom: 6px; background: #11181f; }
    .target.active { outline: 2px solid #4fb286; background: #0f1d19; }
    .target.picked { border-left: 4px solid #4fb286; }
    .target.empty { border-left: 4px solid #56616b; }
    .mono { font-family: monospace; font-size: 12px; }
    .fit-previews { min-height: 0; flex: 1; display: flex; align-items: center; justify-content: center; }
    .fit-preview { width: 100%; height: auto; max-height: 100%; min-height: 120px; object-fit: contain; border-radius: 8px; background: #0d1217; display: block; }
    .verdict-controls { display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap; }
    .verdict-controls button { flex: 1; min-width: 0; padding: 8px 10px; border: 0; border-radius: 6px; cursor: pointer; background: #34424f; color: #e8eef2; }
    .verdict-controls button.active { outline: 2px solid #f4f7fa; }
    .verdict-good.active { background: #4fb286; color: #08110d; }
    .verdict-caution.active { background: #e2a72e; color: #221500; }
    .verdict-bad.active { background: #e25c5c; color: #200707; }
    @media (max-width: 1100px) {
      .layout { grid-template-rows: auto auto; height: auto; min-height: 100vh; }
      .top-row, .bottom-row { grid-template-columns: minmax(0, 1fr); }
      .targets-panel { min-height: 320px; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <div class="top-row">
      <div class="panel image-panel">
        <h2>Image</h2>
        <div class="status" id="sceneSummary">Loading scene...</div>
        <div class="visual-stage">
          <canvas id="imageCanvas"></canvas>
        </div>
      </div>
      <div class="panel fit-panel">
        <h2>Overlay</h2>
        <div class="status" id="fitStatus">Fit feedback unavailable.</div>
        <div class="fit-previews">
          <img class="fit-preview" id="overlayPreview" alt="Overlay preview" hidden>
        </div>
        <div class="verdict-controls">
          <button id="verdictGoodBtn" class="verdict-good">Good</button>
          <button id="verdictCautionBtn" class="verdict-caution">Usable with caution</button>
          <button id="verdictBadBtn" class="verdict-bad">Bad</button>
        </div>
      </div>
    </div>
    <div class="bottom-row">
      <div class="panel cloud-panel">
        <h2>LiDAR</h2>
        <div class="status" id="cloudStatus">Loading point cloud...</div>
        <div id="cloud"></div>
      </div>
    <div class="panel targets-panel">
      <h2>Targets</h2>
      <div class="status" id="studyStatus">Timer: 00:00 | Replacements: 0 | Clears: 0</div>
      <div class="status" id="pickStatus">No active target selected.</div>
      <div class="mono" id="candidatePoint">Candidate point: none</div>
      <div class="controls">
        <button class="secondary" id="timerBtn">Start Timing</button>
        <button class="secondary" id="resetTimerBtn">Reset Timer</button>
        <button class="primary" id="savePickBtn">Save Pick</button>
        <button class="warn" id="replacePickBtn">Replace Pick</button>
        <button class="danger" id="clearPickBtn">Clear Pick</button>
        <button class="secondary" id="exportBtn">Export Picks</button>
        <button class="secondary" id="computeFitBtn">Compute Fit</button>
        </div>
        <div id="targetList"></div>
      </div>
    </div>
  </div>

  <script type="module">
    import * as THREE from 'three';
    import { TrackballControls } from 'three/addons/controls/TrackballControls.js';

    let scene = null;
    let picks = [];
    let activeIndex = null;
    let candidatePoint = null;
    let statusFlash = '';
    let imageElement = new Image();
    let pointcloud = null;
    let canvas = document.getElementById('imageCanvas');
    let ctx = canvas.getContext('2d');
    const sceneSummary = document.getElementById('sceneSummary');
    const cloudStatus = document.getElementById('cloudStatus');
    const pickStatus = document.getElementById('pickStatus');
    const studyStatus = document.getElementById('studyStatus');
    const candidatePointEl = document.getElementById('candidatePoint');
    const targetListEl = document.getElementById('targetList');
    const cloudRoot = document.getElementById('cloud');
    const fitStatus = document.getElementById('fitStatus');
    const overlayPreview = document.getElementById('overlayPreview');

    let cloudScene = null;
    let cloudCamera = null;
    let cloudRenderer = null;
    let cloudControls = null;
    let cloudRaycaster = null;
    let cloudMouse = null;
    let cloudPoints = null;
    let pickedPointsObject = null;
    let candidatePointsObject = null;
    let pointPositions = null;
    let pointerDownAt = null;
    let pointPickThreshold = 0.12;
    let maxPickScreenDistancePx = 18;
    let fitState = null;
    let sessionState = null;
    let timerIntervalId = null;
    let clientTimerAnchorMs = null;
    let clientTimerBaseSeconds = 0;

    async function fetchJson(url, options) {
      const resp = await fetch(url, options);
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || resp.statusText);
      }
      return resp.json();
    }

    function formatElapsed(seconds) {
      const total = Math.max(0, Math.floor(seconds));
      const mins = Math.floor(total / 60);
      const secs = total % 60;
      return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    }

    function currentDisplayedElapsedSeconds() {
      if (!sessionState) return 0;
      if (!sessionState.timing_running || clientTimerAnchorMs === null) {
        return sessionState.elapsed_seconds_current || 0;
      }
      return clientTimerBaseSeconds + Math.max((Date.now() - clientTimerAnchorMs) / 1000.0, 0);
    }

    function updateStudyStatus() {
      if (!sessionState) {
        studyStatus.textContent = 'Timer: 00:00 | Replacements: 0 | Clears: 0';
        return;
      }
      studyStatus.textContent =
        `Timer: ${formatElapsed(currentDisplayedElapsedSeconds())} | ` +
        `Replacements: ${sessionState.replacement_count || 0} | ` +
        `Clears: ${sessionState.clear_count || 0}`;
      document.getElementById('timerBtn').textContent = sessionState.timing_running ? 'Stop Timing' : 'Start Timing';
      const verdict = sessionState.verdict || null;
      document.getElementById('verdictGoodBtn').classList.toggle('active', verdict === 'good');
      document.getElementById('verdictCautionBtn').classList.toggle('active', verdict === 'usable with caution');
      document.getElementById('verdictBadBtn').classList.toggle('active', verdict === 'bad');
    }

    function drawImageCanvas() {
      if (!imageElement.complete || !imageElement.naturalWidth) return;
      canvas.width = imageElement.naturalWidth;
      canvas.height = imageElement.naturalHeight;
      ctx.drawImage(imageElement, 0, 0);

      picks.forEach((pick) => {
        const [u, v] = pick.image_uv;
        ctx.beginPath();
        ctx.arc(u, v, pick.index === activeIndex ? 9 : 6, 0, Math.PI * 2);
        ctx.strokeStyle = pick.status === 'picked' ? '#4fb286' : '#e25c5c';
        ctx.lineWidth = pick.index === activeIndex ? 3 : 2;
        ctx.stroke();
        ctx.fillStyle = '#f4f7fa';
        ctx.font = '14px sans-serif';
        ctx.fillText(String(pick.index), u + 8, v - 8);
      });

      if (fitState && fitState.ready && Array.isArray(fitState.fit_projected_uv)) {
        const projected = fitState.fit_projected_uv;
        const reference = fitState.fit_reference_uv || [];
        for (let i = 0; i < projected.length; i += 1) {
          const uvProj = projected[i];
          if (!uvProj || uvProj.length < 2) continue;
          const uvRef = reference[i];
          if (uvRef && uvRef.length >= 2) {
            ctx.beginPath();
            ctx.moveTo(uvRef[0], uvRef[1]);
            ctx.lineTo(uvProj[0], uvProj[1]);
            ctx.strokeStyle = 'rgba(255,230,109,0.8)';
            ctx.lineWidth = 1.5;
            ctx.stroke();
          }
          ctx.beginPath();
          ctx.moveTo(uvProj[0] - 5, uvProj[1] - 5);
          ctx.lineTo(uvProj[0] + 5, uvProj[1] + 5);
          ctx.moveTo(uvProj[0] - 5, uvProj[1] + 5);
          ctx.lineTo(uvProj[0] + 5, uvProj[1] - 5);
          ctx.strokeStyle = '#ffe66d';
          ctx.lineWidth = 2.5;
          ctx.stroke();
        }
      }
    }

    function renderTargets() {
      targetListEl.innerHTML = '';
      picks.forEach((pick) => {
        const div = document.createElement('div');
        div.className = `target ${pick.status} ${pick.index === activeIndex ? 'active' : ''}`;
        const lasText = pick.las_xyz ? pick.las_xyz.map(v => v.toFixed(3)).join(', ') : 'not picked';
        div.innerHTML = `<strong>#${pick.index}</strong> (${pick.image_uv[0].toFixed(1)}, ${pick.image_uv[1].toFixed(1)})<br><span class="mono">${lasText}</span>`;
        div.onclick = async () => {
          activeIndex = pick.index;
          statusFlash = 'click the matching LiDAR point to save';
          await fetchJson('/api/session', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({selected_target_index: activeIndex}),
          });
          updateStatus();
          renderTargets();
          drawImageCanvas();
          updatePointCloudOverlays();
        };
        targetListEl.appendChild(div);
      });
    }

    function updateStatus() {
      const active = picks.find(p => p.index === activeIndex);
      sceneSummary.textContent = `${scene.scene_label} | targets: ${picks.length} | picked: ${picks.filter(p => p.status === 'picked').length}`;
      cloudStatus.textContent = pointcloud
        ? `Cloud: ${pointcloud.source_kind} (${pointcloud.display_source}) | points rendered: ${pointcloud.point_count} / ${pointcloud.original_point_count}`
        : 'Points rendered: ...';
      if (!active) {
        pickStatus.textContent = 'No active target selected.';
      } else {
        pickStatus.textContent = `Active target #${active.index} at (${active.image_uv[0].toFixed(1)}, ${active.image_uv[1].toFixed(1)})`;
      }
      if (statusFlash) {
        pickStatus.textContent += ` | ${statusFlash}`;
      }
      candidatePointEl.textContent = candidatePoint
        ? `Candidate point: ${candidatePoint.map(v => v.toFixed(4)).join(', ')}`
        : 'Candidate point: none';
      updateStudyStatus();
      updateFitSection();
    }

    function updateFitSection() {
      if (!scene) {
        fitStatus.textContent = 'Fit feedback unavailable for this scene.';
        overlayPreview.hidden = true;
        drawImageCanvas();
        return;
      }
      if (!fitState || !fitState.ready) {
        const picked = scene.capabilities.can_run_fit_feedback
          ? picks.filter(p => p.status === 'picked' && p.txt_xyz).length
          : picks.filter(p => p.status === 'picked').length;
        fitStatus.textContent = scene.capabilities.can_run_fit_feedback
          ? `Fit pending. Need at least 6 matched correspondence points. Current: ${picked}.`
          : `No .cyl yet. Need at least 6 picked points, then Compute Fit with an init .cyl. Current: ${picked}.`;
        overlayPreview.hidden = true;
        drawImageCanvas();
        return;
      }
      if (fitState.mode === 'generated_cyl') {
        fitStatus.textContent = `Fit RMSE: ${fitState.fit_rmse_total.toFixed(3)} px | Init .cyl: ${fitState.init_cyl} | Fitted .cyl: ${fitState.fitted_cyl}`;
      } else {
        fitStatus.textContent = `Fit RMSE: ${fitState.fit_rmse_total.toFixed(3)} px | .cyl RMSE: ${fitState.cyl_verify_rmse_total.toFixed(3)} px`;
      }
      overlayPreview.src = `/api/artifacts/overlay_preview.png?ts=${encodeURIComponent(fitState.updated_at)}`;
      overlayPreview.hidden = false;
      drawImageCanvas();
    }

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
      const material = new THREE.PointsMaterial({
        size: 2.0,
        sizeAttenuation: false,
        vertexColors: true,
        transparent: true,
        opacity: 0.92,
      });
      return new THREE.Points(geometry, material);
    }

    function updatePointCloudOverlays() {
      if (!cloudScene) return;
      const picked = picks.filter(p => p.las_xyz);
      const pickedGeometry = buildPointGeometry(
        picked.map(p => p.las_xyz[0]),
        picked.map(p => p.las_xyz[1]),
        picked.map(p => p.las_xyz[2]),
      );
      if (pickedPointsObject) {
        cloudScene.remove(pickedPointsObject);
        pickedPointsObject.geometry.dispose();
        pickedPointsObject.material.dispose();
      }
      pickedPointsObject = new THREE.Points(
        pickedGeometry,
        new THREE.PointsMaterial({
          size: 9,
          sizeAttenuation: false,
          color: '#ff5d73',
          transparent: true,
          opacity: 0.95,
        }),
      );
      cloudScene.add(pickedPointsObject);

      const active = picks.find(p => p.index === activeIndex && p.las_xyz);
      if (active) {
        const activeGeometry = buildPointGeometry([active.las_xyz[0]], [active.las_xyz[1]], [active.las_xyz[2]]);
        const activeObject = new THREE.Points(
          activeGeometry,
          new THREE.PointsMaterial({
            size: 12,
            sizeAttenuation: false,
            color: '#ffd166',
            transparent: true,
            opacity: 1.0,
          }),
        );
        pickedPointsObject.add(activeObject);
      }

      if (candidatePointsObject) {
        cloudScene.remove(candidatePointsObject);
        candidatePointsObject.geometry.dispose();
        candidatePointsObject.material.dispose();
      }
      candidatePointsObject = new THREE.Points(
        buildPointGeometry(
          candidatePoint ? [candidatePoint[0]] : [],
          candidatePoint ? [candidatePoint[1]] : [],
          candidatePoint ? [candidatePoint[2]] : [],
        ),
        new THREE.PointsMaterial({
          size: 11,
          sizeAttenuation: false,
          color: '#7cfcff',
          transparent: true,
          opacity: 1.0,
        }),
      );
      cloudScene.add(candidatePointsObject);
    }

    function onCloudResize() {
      if (!cloudRenderer || !cloudCamera) return;
      const width = cloudRoot.clientWidth;
      const height = Math.max(cloudRoot.clientHeight, 400);
      cloudCamera.aspect = width / height;
      cloudCamera.updateProjectionMatrix();
      cloudRenderer.setSize(width, height, false);
    }

    function animateCloud() {
      requestAnimationFrame(animateCloud);
      if (cloudControls) cloudControls.update();
      if (cloudRenderer && cloudScene && cloudCamera) {
        cloudRenderer.render(cloudScene, cloudCamera);
      }
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
        if (!Number.isFinite(projected.x) || !Number.isFinite(projected.y) || !Number.isFinite(projected.z)) {
          continue;
        }
        if (projected.z < -1 || projected.z > 1) {
          continue;
        }
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
      if (bestIdx < 0) {
        return null;
      }
      return {
        index: bestIdx,
        distancePx: Math.sqrt(bestDist2),
        point: [
          pointPositions[3 * bestIdx],
          pointPositions[3 * bestIdx + 1],
          pointPositions[3 * bestIdx + 2],
        ],
      };
    }

    async function handlePointPick(event) {
      if (!cloudPoints) return;
      if (!event.ctrlKey) {
        statusFlash = 'hold Ctrl and click to pick a LiDAR point';
        updateStatus();
        return;
      }
      const hit = findNearestScreenPoint(event, maxPickScreenDistancePx);
      if (!hit) {
        statusFlash = 'no point close enough to cursor';
        updateStatus();
        return;
      }
      candidatePoint = [
        hit.point[0],
        hit.point[1],
        hit.point[2],
      ];
      statusFlash = activeIndex === null
        ? 'candidate selected; select an image target first'
        : `candidate selected for #${activeIndex} with Ctrl+click (${hit.distancePx.toFixed(1)} px)`;
      updatePointCloudOverlays();
      updateStatus();
      if (activeIndex !== null) {
        await saveCurrentCandidate(false);
      }
    }

    function setRotationCenterFromEvent(event) {
      if (!cloudPoints) return;
      const hit = findNearestScreenPoint(event, maxPickScreenDistancePx);
      if (!hit) {
        statusFlash = 'no point found to use as rotation center';
        updateStatus();
        return;
      }
      const center = new THREE.Vector3(
        hit.point[0],
        hit.point[1],
        hit.point[2],
      );
      const offset = new THREE.Vector3().subVectors(cloudCamera.position, cloudControls.target);
      cloudControls.target.copy(center);
      cloudCamera.position.copy(center.clone().add(offset));
      cloudCamera.updateProjectionMatrix();
      statusFlash = 'rotation center moved';
      cloudControls.update();
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
      cloudControls.noRoll = false;

      cloudRaycaster = new THREE.Raycaster();
      cloudMouse = new THREE.Vector2();
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
        await handlePointPick(event);
      });
      window.addEventListener('resize', onCloudResize);
      onCloudResize();
      updatePointCloudOverlays();
      animateCloud();
      updateStatus();
    }

    async function loadScene() {
      scene = await fetchJson('/api/scene');
      sessionState = scene.session;
      clientTimerBaseSeconds = sessionState.elapsed_seconds || 0;
      clientTimerAnchorMs = sessionState.timing_running ? Date.now() : null;
      const picksPayload = await fetchJson('/api/picks');
      picks = picksPayload.picks;
      activeIndex = scene.session ? scene.session.selected_target_index : (picks.length ? picks[0].index : null);
      imageElement.onload = () => {
        drawImageCanvas();
        renderTargets();
        updateStatus();
      };
      imageElement.src = `/api/image?scene_key=${encodeURIComponent(scene.scene_key)}&ts=${encodeURIComponent(Date.now())}`;
      pointcloud = await fetchJson('/api/pointcloud');
      renderPointCloud();
      await refreshFit();
      if (timerIntervalId === null) {
        timerIntervalId = window.setInterval(async () => {
          if (!sessionState || !sessionState.timing_running) return;
          sessionState = await fetchJson('/api/session');
          clientTimerBaseSeconds = sessionState.elapsed_seconds || 0;
          if (sessionState.timing_running && clientTimerAnchorMs === null) {
            clientTimerAnchorMs = Date.now();
          }
          if (!sessionState.timing_running) {
            clientTimerAnchorMs = null;
          }
          updateStudyStatus();
        }, 1000);
      }
    }

    async function refreshPicks() {
      const picksPayload = await fetchJson('/api/picks');
      picks = picksPayload.picks;
      sessionState = await fetchJson('/api/session');
      renderTargets();
      drawImageCanvas();
      updatePointCloudOverlays();
      updateStatus();
      if (scene.capabilities.can_run_fit_feedback) {
        await refreshFit();
      }
    }

    async function refreshFit() {
      fitState = await fetchJson('/api/fit');
      updateFitSection();
    }

    async function saveCurrentCandidate(forceReplace = false) {
      if (activeIndex === null || candidatePoint === null) return false;
      const active = picks.find(p => p.index === activeIndex);
      if (!active) return false;
      if (active.las_xyz && !forceReplace) {
        const ok = window.confirm(`Target #${active.index} already has a pick. Replace it?`);
        if (!ok) return false;
      }
      await fetchJson(`/api/picks/${activeIndex}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({las_xyz: candidatePoint}),
      });
      statusFlash = `saved pick for #${activeIndex}`;
      await refreshPicks();
      const pickedFitCount = picks.filter(p => p.status === 'picked' && p.txt_xyz).length;
      if (scene.capabilities.can_run_fit_feedback && pickedFitCount >= 6) {
        await computeFit();
      }
      return true;
    }

    async function computeFit() {
      fitStatus.textContent = 'Computing fit preview...';
      let payload = {};
      if (!scene.capabilities.can_run_fit_feedback) {
        const defaultInitCyl = scene.source_paths ? (scene.source_paths.default_init_cyl || '') : '';
        const initCyl = window.prompt('Path to initial .cyl for fitting this scene:', defaultInitCyl);
        if (!initCyl) {
          statusFlash = 'fit cancelled: initial .cyl is required';
          updateStatus();
          return;
        }
        const defaultOptMode = scene.source_paths ? (scene.source_paths.default_fit_opt_mode || 'all') : 'all';
        payload = {init_cyl_path: initCyl, opt_mode: defaultOptMode};
      }
      fitState = await fetchJson('/api/fit', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload),
      });
      updateFitSection();
    }

    canvas.addEventListener('click', async (event) => {
      if (event.altKey) {
        const rect = canvas.getBoundingClientRect();
        const sx = canvas.width / rect.width;
        const sy = canvas.height / rect.height;
        const x = (event.clientX - rect.left) * sx;
        const y = (event.clientY - rect.top) * sy;
        let nearest = null;
        let bestDist = Infinity;
        for (const pick of picks) {
          const dx = pick.image_uv[0] - x;
          const dy = pick.image_uv[1] - y;
          const d = Math.sqrt(dx * dx + dy * dy);
          if (d < bestDist) {
            bestDist = d;
            nearest = pick.index;
          }
        }
        if (nearest !== null && bestDist <= 18) {
          await fetchJson(`/api/targets/${nearest}`, {method: 'DELETE'});
          if (activeIndex === nearest) {
            candidatePoint = null;
          }
          statusFlash = `cleared image target #${nearest}`;
          await refreshPicks();
        }
        return;
      }
      if (scene.capabilities.has_corresp_txt) {
        const rect = canvas.getBoundingClientRect();
        const sx = canvas.width / rect.width;
        const sy = canvas.height / rect.height;
        const x = (event.clientX - rect.left) * sx;
        const y = (event.clientY - rect.top) * sy;
        let nearest = null;
        let bestDist = Infinity;
        for (const pick of picks) {
          const dx = pick.image_uv[0] - x;
          const dy = pick.image_uv[1] - y;
          const d = Math.sqrt(dx * dx + dy * dy);
          if (d < bestDist) {
            bestDist = d;
            nearest = pick.index;
          }
        }
        if (nearest !== null && bestDist <= 18) {
          activeIndex = nearest;
          statusFlash = 'click the matching LiDAR point to save';
          await fetchJson('/api/session', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({selected_target_index: activeIndex}),
          });
          renderTargets();
          drawImageCanvas();
          updatePointCloudOverlays();
          updateStatus();
        }
        return;
      }

      const rect = canvas.getBoundingClientRect();
      const sx = canvas.width / rect.width;
      const sy = canvas.height / rect.height;
      const image_uv = [(event.clientX - rect.left) * sx, (event.clientY - rect.top) * sy];
      const created = await fetchJson('/api/targets', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({image_uv}),
      });
      activeIndex = created.index;
      await refreshPicks();
    });

    document.getElementById('savePickBtn').onclick = async () => {
      await saveCurrentCandidate(false);
    };

    document.getElementById('replacePickBtn').onclick = async () => {
      await saveCurrentCandidate(true);
    };

    document.getElementById('clearPickBtn').onclick = async () => {
      if (activeIndex === null) return;
      await fetchJson(`/api/picks/${activeIndex}`, {method: 'DELETE'});
      candidatePoint = null;
      statusFlash = `cleared pick for #${activeIndex}`;
      await refreshPicks();
    };

    document.getElementById('exportBtn').onclick = async () => {
      const payload = await fetchJson('/api/export', {method: 'POST'});
      alert(`Exported picks to ${payload.export_csv}\nGenerated correspondences: ${payload.generated_corresp_txt}`);
    };

    document.getElementById('computeFitBtn').onclick = async () => {
      await computeFit();
    };

    document.getElementById('timerBtn').onclick = async () => {
      if (sessionState && sessionState.timing_running) {
        const elapsed = currentDisplayedElapsedSeconds();
        sessionState = await fetchJson('/api/timer/stop', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({elapsed_seconds: elapsed}),
        });
        clientTimerBaseSeconds = sessionState.elapsed_seconds || elapsed;
        clientTimerAnchorMs = null;
      } else {
        sessionState = await fetchJson('/api/timer/start', {method: 'POST'});
        clientTimerBaseSeconds = sessionState.elapsed_seconds || 0;
        clientTimerAnchorMs = Date.now();
      }
      updateStudyStatus();
    };

    document.getElementById('resetTimerBtn').onclick = async () => {
      sessionState = await fetchJson('/api/timer/reset', {method: 'POST'});
      clientTimerBaseSeconds = 0;
      clientTimerAnchorMs = null;
      updateStudyStatus();
    };

    document.getElementById('verdictGoodBtn').onclick = async () => {
      sessionState = await fetchJson('/api/verdict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({verdict: 'good'}),
      });
      updateStudyStatus();
    };

    document.getElementById('verdictCautionBtn').onclick = async () => {
      sessionState = await fetchJson('/api/verdict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({verdict: 'usable with caution'}),
      });
      updateStudyStatus();
    };

    document.getElementById('verdictBadBtn').onclick = async () => {
      sessionState = await fetchJson('/api/verdict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({verdict: 'bad'}),
      });
      updateStudyStatus();
    };

    loadScene().catch((err) => {
      console.error(err);
      sceneSummary.textContent = 'Failed to load scene: ' + err.message;
    });
  </script>
</body>
</html>
"""

class PickPayload(BaseModel):
    las_xyz: list[float]


class TargetPayload(BaseModel):
    image_uv: list[float]


class SessionPayload(BaseModel):
    selected_target_index: int | None = None


class VerdictPayload(BaseModel):
    verdict: str | None = None


class TimerStopPayload(BaseModel):
    elapsed_seconds: float | None = None


class ComputeFitPayload(BaseModel):
    init_cyl_path: str | None = None
    opt_mode: str = "extr"


def create_app(workspace: SceneWorkspace) -> FastAPI:
    workspace.prepare()
    app = FastAPI(title="Annotation Workspace MVP")
    app.state.workspace = workspace

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(INDEX_HTML)

    @app.get("/api/scene")
    async def get_scene() -> JSONResponse:
        payload = workspace.get_scene_payload()
        payload["session"] = workspace.get_session()
        return JSONResponse(payload)

    @app.get("/api/image")
    async def get_image() -> FileResponse:
        return FileResponse(workspace.image_preview_path)

    @app.get("/api/pointcloud")
    async def get_pointcloud() -> JSONResponse:
        try:
            payload = workspace.get_pointcloud_payload()
        except FileNotFoundError:
            return JSONResponse(
                {
                    "ready": False,
                    "preprocessing": workspace.get_preprocess_status(),
                },
                status_code=409,
            )
        return JSONResponse(payload)

    @app.get("/api/picks")
    async def get_picks() -> JSONResponse:
        return JSONResponse(workspace.get_picks())

    @app.get("/api/session")
    async def get_session() -> JSONResponse:
        return JSONResponse(workspace.get_session())

    @app.get("/api/fit")
    async def get_fit() -> JSONResponse:
        return JSONResponse(workspace.get_fit_status())

    @app.post("/api/fit")
    async def compute_fit(payload: ComputeFitPayload | None = None) -> JSONResponse:
        try:
            result = workspace.compute_fit(
                init_cyl_path=None if payload is None else payload.init_cyl_path,
                opt_mode="extr" if payload is None else payload.opt_mode,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(result)

    @app.get("/api/artifacts/{name}")
    async def get_artifact(name: str) -> FileResponse:
        allowed = {
            "overlay_preview.png": workspace.overlay_preview_path,
            "reprojection_preview.png": workspace.reprojection_preview_path,
            "cyl_verification_overlay.png": workspace.cyl_verification_overlay_path,
        }
        path = allowed.get(name)
        if path is None or not path.exists():
            raise HTTPException(status_code=404, detail=f"Artifact not found: {name}")
        return FileResponse(path)

    @app.post("/api/picks/{index}")
    async def upsert_pick(index: int, payload: PickPayload) -> JSONResponse:
        try:
            result = workspace.upsert_pick(index, payload.las_xyz)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(result)

    @app.delete("/api/picks/{index}")
    async def clear_pick(index: int) -> JSONResponse:
        try:
            result = workspace.clear_pick(index)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(result)

    @app.post("/api/targets")
    async def add_target(payload: TargetPayload) -> JSONResponse:
        result = workspace.add_target(payload.image_uv)
        return JSONResponse(result)

    @app.delete("/api/targets/{index}")
    async def delete_target(index: int) -> JSONResponse:
        try:
            result = workspace.delete_target(index)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(result)

    @app.post("/api/session")
    async def update_session(payload: SessionPayload) -> JSONResponse:
        result = workspace.update_session(payload.selected_target_index)
        return JSONResponse(result)

    @app.post("/api/timer/start")
    async def start_timer() -> JSONResponse:
        return JSONResponse(workspace.start_timer())

    @app.post("/api/timer/stop")
    async def stop_timer(payload: TimerStopPayload | None = None) -> JSONResponse:
        elapsed_override = None if payload is None else payload.elapsed_seconds
        return JSONResponse(workspace.stop_timer(elapsed_override))

    @app.post("/api/timer/reset")
    async def reset_timer() -> JSONResponse:
        return JSONResponse(workspace.reset_timer())

    @app.post("/api/verdict")
    async def set_verdict(payload: VerdictPayload) -> JSONResponse:
        try:
            return JSONResponse(workspace.set_verdict(payload.verdict))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/export")
    async def export_picks() -> JSONResponse:
        export_csv = workspace.export_picks_csv()
        generated_corresp = workspace.export_generated_corresp_txt()
        return JSONResponse(
            {
                "export_csv": str(export_csv),
                "generated_corresp_txt": str(generated_corresp),
            }
        )

    return app


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Launch the single-scene annotation workspace.")
    ap.add_argument("--collection", required=True)
    ap.add_argument("--path-name", required=True)
    ap.add_argument("--step", required=True, type=int)
    ap.add_argument("--force-generated-cyl-mode", action="store_true")
    ap.add_argument("--use-reference-targets-in-generated-mode", action="store_true")
    ap.add_argument("--workspace-variant")
    ap.add_argument("--default-init-cyl")
    ap.add_argument("--default-fit-opt-mode", default="all")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", default=8000, type=int)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    workspace = SceneWorkspace(
        args.collection,
        args.path_name,
        args.step,
        force_generated_cyl_mode=args.force_generated_cyl_mode,
        use_reference_targets_in_generated_mode=args.use_reference_targets_in_generated_mode,
        workspace_variant=args.workspace_variant,
        default_init_cyl_path=args.default_init_cyl,
        default_fit_opt_mode=args.default_fit_opt_mode,
    )
    app = create_app(workspace)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
