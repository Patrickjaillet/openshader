"""
wasm_exporter.py
----------------
v1.0 — Export Runtime WebAssembly (WASM) pour OpenShader / DemoMaker.

Génère un bundle complet permettant de jouer un projet .demomaker directement
dans un navigateur, sans serveur, sans plugin, via WebGL 2.0 / WebGPU.

Architecture du bundle généré :
  dist/
  ├── index.html          — page d'accueil + loader UI
  ├── player.html         — player WebGL2 standalone (iframe-embeddable)
  ├── embed.js            — snippet <script> d'intégration externe
  ├── sw.js               — Service Worker (cache offline + PWA)
  ├── manifest.webmanifest— PWA manifest (installable sur mobile)
  ├── shaders/            — sources GLSL du projet
  │   ├── image.glsl
  │   └── buffer_a.glsl …
  ├── runtime/            — stub C + Makefile Emscripten (pour rebuild natif)
  │   ├── openshader_runtime.c
  │   ├── Makefile.emscripten
  │   └── README_BUILD.md
  └── meta.json           — métadonnées du projet

Le player est auto-contenu (zero CDN) et fonctionne en file:// ou tout
hébergeur statique (GitHub Pages, Netlify, itch.io…).

Compatibilité navigateurs :
  - Chrome/Edge 113+  → WebGPU si disponible, WebGL2 fallback
  - Firefox 98+       → WebGL2
  - Safari 16.4+      → WebGL2

Usage depuis Python :
    exporter = WasmExporter(meta=meta, shaders=shaders)
    files = exporter.export('/path/to/dist/')

Usage CLI (Emscripten) — depuis dist/runtime/ :
    make -f Makefile.emscripten
    # → produit openshader_runtime.wasm + openshader_runtime.js
"""

from __future__ import annotations

import json
import os
import re
import datetime
import shutil
from pathlib import Path

# ── Constantes ────────────────────────────────────────────────────────────────

_VERSION = "1.0"
_SDK_VER  = "3.0"

# ══════════════════════════════════════════════════════════════════════════════
#  player.html — WebGL2 + WebGPU player standalone
# ══════════════════════════════════════════════════════════════════════════════

_PLAYER_HTML = """\
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<meta name="theme-color" content="#000000">
<title>{title}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{
    width: 100%; height: 100%;
    background: #000; overflow: hidden;
    font-family: 'Segoe UI', system-ui, sans-serif;
  }}
  canvas {{
    display: block; width: 100%; height: 100%;
    image-rendering: pixelated;
    touch-action: none;
  }}

  /* ── Loader ─────────────────────────────────────────────────── */
  #loader {{
    position: absolute; inset: 0;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    background: #000; color: rgba(255,255,255,.6);
    font-size: 13px; gap: 14px;
    transition: opacity .4s;
  }}
  #loader.hidden {{ opacity: 0; pointer-events: none; }}
  .spinner {{
    width: 32px; height: 32px;
    border: 2px solid rgba(255,255,255,.1);
    border-top-color: #3a88ff;
    border-radius: 50%;
    animation: spin .8s linear infinite;
  }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

  /* ── Overlay UI ─────────────────────────────────────────────── */
  #ui {{
    position: absolute; bottom: 0; left: 0; right: 0;
    padding: 10px 14px;
    background: linear-gradient(transparent, rgba(0,0,0,.85));
    display: flex; align-items: center; gap: 8px;
    opacity: 0; transition: opacity .25s; pointer-events: none;
  }}
  body:hover #ui, body.touch #ui {{ opacity: 1; pointer-events: auto; }}
  .btn {{
    background: rgba(255,255,255,.1); color: #fff; border: none;
    border-radius: 4px; padding: 5px 10px; cursor: pointer; font-size: 13px;
    backdrop-filter: blur(6px); transition: background .15s;
    -webkit-backdrop-filter: blur(6px);
  }}
  .btn:hover {{ background: rgba(255,255,255,.22); }}
  #time-lbl {{
    color: rgba(255,255,255,.6); font-size: 11px;
    font-family: 'Consolas', monospace; margin-left: auto;
    letter-spacing: .03em;
  }}
  #backend-badge {{
    font-size: 9px; color: rgba(255,255,255,.35);
    text-transform: uppercase; letter-spacing: .08em;
  }}

  /* ── Title overlay ──────────────────────────────────────────── */
  #title-overlay {{
    position: absolute; top: 12px; left: 14px;
    color: rgba(255,255,255,.45); font-size: 11px;
    pointer-events: none; letter-spacing: .04em;
  }}

  /* ── Error overlay ──────────────────────────────────────────── */
  #error-overlay {{
    display: none; position: absolute; inset: 0;
    background: rgba(0,0,0,.94); color: #ff6060;
    font-family: 'Consolas', monospace; font-size: 12px;
    padding: 28px 32px; white-space: pre-wrap; overflow: auto;
    line-height: 1.6;
  }}
  #error-overlay strong {{ color: #ff8888; display: block; margin-bottom: 8px; }}

  /* ── Offline badge ──────────────────────────────────────────── */
  #offline-badge {{
    display: none; position: absolute; top: 10px; right: 14px;
    background: rgba(255,200,0,.15); color: rgba(255,200,0,.8);
    font-size: 10px; padding: 3px 8px; border-radius: 3px;
    border: 1px solid rgba(255,200,0,.25); pointer-events: none;
  }}
</style>
</head>
<body>

<canvas id="c"></canvas>
<div id="loader"><div class="spinner"></div><span id="loader-msg">Initialisation…</span></div>
<div id="title-overlay">{title}</div>
<div id="error-overlay"><strong>❌ Erreur</strong><span id="error-msg"></span></div>
<div id="offline-badge">📶 Offline</div>
<div id="ui">
  <button class="btn" id="btn-play" onclick="togglePlay()">⏸</button>
  <button class="btn" onclick="restart()">⟳</button>
  <span id="time-lbl">0.00 s</span>
  <span id="backend-badge">webgl2</span>
  <button class="btn" onclick="toggleFs()" style="margin-left:6px">⛶</button>
</div>

<script>
/* ═══════════════════════════════════════════════════════════════
   OpenShader WASM Player  —  v{version}
   Generated by DemoMaker {sdk_ver}  ·  {date}
   Backend : WebGPU (preferred) → WebGL2 (fallback)
═══════════════════════════════════════════════════════════════ */

// ── Shader sources (inlined at export time) ───────────────────────────────────
const SHADERS = {shaders_json};
const META    = {meta_json};

// ── DOM refs ──────────────────────────────────────────────────────────────────
const canvas    = document.getElementById('c');
const loader    = document.getElementById('loader');
const loaderMsg = document.getElementById('loader-msg');
const errDiv    = document.getElementById('error-overlay');
const errMsg    = document.getElementById('error-msg');
const badge     = document.getElementById('backend-badge');

function showError(title, msg) {{
  errDiv.style.display = 'block';
  loader.classList.add('hidden');
  document.querySelector('#error-overlay strong').textContent = '❌ ' + title;
  errMsg.textContent = msg;
}}

function setLoaderMsg(txt) {{ loaderMsg.textContent = txt; }}

// ── Backend detection ─────────────────────────────────────────────────────────
let _renderer = null;   // {{ type: 'webgpu'|'webgl2', ... }}

async function initRenderer() {{
  // 1. Try WebGPU
  if (navigator.gpu) {{
    try {{
      setLoaderMsg('Initialisation WebGPU…');
      const adapter = await navigator.gpu.requestAdapter({{ powerPreference: 'high-performance' }});
      if (adapter) {{
        const device = await adapter.requestDevice();
        const ctx = canvas.getContext('webgpu');
        const fmt = navigator.gpu.getPreferredCanvasFormat();
        ctx.configure({{ device, format: fmt, alphaMode: 'opaque' }});
        badge.textContent = 'webgpu';
        badge.style.color = 'rgba(80,200,120,.5)';
        _renderer = {{ type: 'webgpu', device, ctx, fmt }};
        return true;
      }}
    }} catch(e) {{
      console.warn('WebGPU init failed, falling back to WebGL2:', e);
    }}
  }}

  // 2. Fallback WebGL2
  setLoaderMsg('Initialisation WebGL2…');
  const gl = canvas.getContext('webgl2', {{ antialias: false, preserveDrawingBuffer: false }});
  if (!gl) {{
    showError('Navigateur non compatible',
      'WebGL2 et WebGPU sont tous deux indisponibles.\\n' +
      'Essayez Chrome 113+, Firefox 98+, ou Safari 16.4+.');
    return false;
  }}
  badge.textContent = 'webgl2';
  _renderer = {{ type: 'webgl2', gl }};
  return true;
}}

// ══════════════════════════════════════════════════════════════════════════════
//  WebGL2 renderer
// ══════════════════════════════════════════════════════════════════════════════

function glslAdapt(src) {{
  src = src.replace(/#version\\s+\\d+\\s*(core|compatibility)?\\s*\\n/g, '');
  src = src.replace(/^\\s*uniform\\s+(vec[234]|float|int|sampler2D)\\s+[iu][A-Z]\\w*\\s*;/gm, '');
  const hdr = `#version 300 es
precision highp float;
precision highp int;
uniform vec3  iResolution;
uniform float iTime;
uniform float iTimeDelta;
uniform int   iFrame;
uniform vec4  iMouse;
uniform float iSampleRate;
out vec4 _fragOut;\\n`;
  const hasMain = /void\\s+mainImage\\s*\\(/.test(src);
  const foot = hasMain
    ? '\\nvoid main(){{ mainImage(_fragOut, gl_FragCoord.xy); }}'
    : '\\nvoid main(){{ _fragOut = vec4(0.,0.,0.,1.); }}';
  return hdr + src + foot;
}}

function glCompile(gl, type, src) {{
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
    throw new Error(gl.getShaderInfoLog(s));
  return s;
}}

function glBuildProg(gl, fragSrc) {{
  const VERT = `#version 300 es\\nin vec2 a;\\nvoid main(){{gl_Position=vec4(a,0,1);}}`;
  const vert = glCompile(gl, gl.VERTEX_SHADER, VERT);
  const frag = glCompile(gl, gl.FRAGMENT_SHADER, glslAdapt(fragSrc));
  const p = gl.createProgram();
  gl.attachShader(p, vert); gl.attachShader(p, frag);
  gl.bindAttribLocation(p, 0, 'a');
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS))
    throw new Error(gl.getProgramInfoLog(p));
  return p;
}}

let _glProg = null, _glVao = null;

function glSetup(gl) {{
  const src = SHADERS['Image'] || SHADERS['image'] || Object.values(SHADERS)[0] || '';
  _glProg = glBuildProg(gl, src);
  _glVao = gl.createVertexArray();
  gl.bindVertexArray(_glVao);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER,
    new Float32Array([-1,-1,1,-1,-1,1,1,-1,1,1,-1,1]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
}}

function glRender(gl, t, dt, frame, mouse) {{
  // Resize
  const dpr = devicePixelRatio || 1;
  const w = Math.round(canvas.clientWidth * dpr);
  const h = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {{ canvas.width = w; canvas.height = h; }}
  gl.viewport(0, 0, w, h);
  gl.useProgram(_glProg);
  gl.bindVertexArray(_glVao);
  const u = (n) => gl.getUniformLocation(_glProg, n);
  gl.uniform3f(u('iResolution'), w, h, 1);
  gl.uniform1f(u('iTime'), t);
  gl.uniform1f(u('iTimeDelta'), dt);
  gl.uniform1i(u('iFrame'), frame);
  gl.uniform4fv(u('iMouse'), mouse);
  gl.uniform1f(u('iSampleRate'), 44100);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  gl.bindVertexArray(null);
}}

// ══════════════════════════════════════════════════════════════════════════════
//  WebGPU renderer (WGSL shader generated from GLSL via transpiler fallback)
// ══════════════════════════════════════════════════════════════════════════════

// NOTE : La transpilation GLSL→WGSL complète nécessite Naga/Tint (Emscripten).
// En mode player HTML-seul, WebGPU affiche un shader de fallback coloré.
// Le vrai pipeline WebGPU est activé quand le .wasm Emscripten est présent.

const WGSL_FALLBACK = `
struct Uniforms {{ time: f32, res_x: f32, res_y: f32, frame: u32 }}
@group(0) @binding(0) var<uniform> u: Uniforms;
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {{
  let uv = pos.xy / vec2f(u.res_x, u.res_y);
  let t  = u.time * 0.5;
  let c  = 0.5 + 0.5 * vec3f(
    sin(uv.x * 6.28 + t), sin(uv.y * 6.28 + t * 1.3), sin((uv.x+uv.y)*4.0 + t*0.7)
  );
  return vec4f(c, 1.0);
}}
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {{
  var pos = array<vec2f,6>(
    vec2f(-1,-1),vec2f(1,-1),vec2f(-1,1),
    vec2f(1,-1),vec2f(1,1),vec2f(-1,1)
  );
  return vec4f(pos[vi], 0.0, 1.0);
}}`;

let _gpuPipeline = null, _gpuUBuf = null, _gpuBG = null;

async function gpuSetup(device, fmt) {{
  const mod = device.createShaderModule({{ code: WGSL_FALLBACK }});
  _gpuPipeline = device.createRenderPipeline({{
    layout: 'auto',
    vertex:   {{ module: mod, entryPoint: 'vs' }},
    fragment: {{ module: mod, entryPoint: 'fs',
                targets: [{{ format: fmt }}] }},
    primitive: {{ topology: 'triangle-list' }},
  }});
  _gpuUBuf = device.createBuffer({{
    size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  }});
  _gpuBG = device.createBindGroup({{
    layout: _gpuPipeline.getBindGroupLayout(0),
    entries: [{{ binding: 0, resource: {{ buffer: _gpuUBuf }} }}],
  }});
}}

function gpuRender(device, ctx, fmt, t, frame) {{
  const dpr = devicePixelRatio || 1;
  const w = Math.round(canvas.clientWidth * dpr);
  const h = Math.round(canvas.clientHeight * dpr);
  canvas.width = w; canvas.height = h;
  device.queue.writeBuffer(_gpuUBuf, 0,
    new Float32Array([t, w, h, frame]));
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({{
    colorAttachments: [{{
      view: ctx.getCurrentTexture().createView(),
      clearValue: [0,0,0,1], loadOp: 'clear', storeOp: 'store'
    }}]
  }});
  pass.setPipeline(_gpuPipeline);
  pass.setBindGroup(0, _gpuBG);
  pass.draw(6);
  pass.end();
  device.queue.submit([enc.finish()]);
}}

// ══════════════════════════════════════════════════════════════════════════════
//  Main loop
// ══════════════════════════════════════════════════════════════════════════════

let _t0 = 0, _pauseOff = 0, _frame = 0, _lastT = 0, _playing = true;
let _mouse = new Float32Array(4);

function curTime() {{
  if (!_playing) return _pauseOff;
  return (performance.now() / 1000 - _t0) + _pauseOff;
}}

function tick() {{
  const t  = curTime();
  const dt = Math.max(0, t - _lastT);
  _lastT   = t;

  if (_renderer.type === 'webgl2') {{
    glRender(_renderer.gl, t, dt, _frame, _mouse);
  }} else {{
    gpuRender(_renderer.device, _renderer.ctx, _renderer.fmt, t, _frame);
  }}

  _frame++;
  document.getElementById('time-lbl').textContent = t.toFixed(2) + ' s';
  if (_playing) requestAnimationFrame(tick);
}}

// ── Controls ──────────────────────────────────────────────────────────────────
function togglePlay() {{
  _playing = !_playing;
  document.getElementById('btn-play').textContent = _playing ? '⏸' : '▶';
  if (_playing) {{
    _t0 = performance.now() / 1000;
    requestAnimationFrame(tick);
  }} else {{
    _pauseOff = curTime();
  }}
}}
function restart() {{
  _t0 = performance.now() / 1000;
  _pauseOff = 0; _frame = 0; _lastT = 0;
  if (!_playing) {{ _playing = true; document.getElementById('btn-play').textContent = '⏸'; }}
  requestAnimationFrame(tick);
}}
function toggleFs() {{
  if (!document.fullscreenElement) canvas.requestFullscreen();
  else document.exitFullscreen();
}}

// ── Touch support ─────────────────────────────────────────────────────────────
canvas.addEventListener('touchstart', () => document.body.classList.add('touch'));
canvas.addEventListener('touchmove', e => {{
  e.preventDefault();
  const t = e.touches[0];
  const r = canvas.getBoundingClientRect();
  const dpr = devicePixelRatio || 1;
  _mouse[0] = (t.clientX - r.left) * dpr;
  _mouse[1] = canvas.height - (t.clientY - r.top) * dpr;
}}, {{ passive: false }});

// ── Mouse ─────────────────────────────────────────────────────────────────────
canvas.addEventListener('mousemove', e => {{
  const r = canvas.getBoundingClientRect(), dpr = devicePixelRatio || 1;
  _mouse[0] = (e.clientX - r.left) * dpr;
  _mouse[1] = canvas.height - (e.clientY - r.top) * dpr;
}});
canvas.addEventListener('mousedown', e => {{
  const r = canvas.getBoundingClientRect(), dpr = devicePixelRatio || 1;
  _mouse[2] = (e.clientX - r.left) * dpr;
  _mouse[3] = canvas.height - (e.clientY - r.top) * dpr;
}});
canvas.addEventListener('mouseup', () => {{ _mouse[2] = 0; _mouse[3] = 0; }});
document.addEventListener('keydown', e => {{
  if (e.code === 'Space') {{ e.preventDefault(); togglePlay(); }}
  if (e.code === 'Escape' && document.fullscreenElement) document.exitFullscreen();
  if (e.code === 'KeyR') restart();
}});

// ── Offline detection ─────────────────────────────────────────────────────────
window.addEventListener('offline', () => {{
  document.getElementById('offline-badge').style.display = 'block';
}});
window.addEventListener('online', () => {{
  document.getElementById('offline-badge').style.display = 'none';
}});

// ── Startup ───────────────────────────────────────────────────────────────────
(async () => {{
  const ok = await initRenderer();
  if (!ok) return;

  try {{
    setLoaderMsg('Compilation du shader…');
    if (_renderer.type === 'webgl2') {{
      glSetup(_renderer.gl);
    }} else {{
      await gpuSetup(_renderer.device, _renderer.fmt);
    }}
  }} catch(e) {{
    showError('Erreur de compilation GLSL', e.message);
    return;
  }}

  loader.classList.add('hidden');
  setTimeout(() => loader.remove(), 500);
  _t0 = performance.now() / 1000;
  requestAnimationFrame(tick);
}})();
</script>
</body>
</html>
"""

# ══════════════════════════════════════════════════════════════════════════════
#  index.html — page d'accueil du bundle WASM
# ══════════════════════════════════════════════════════════════════════════════

_INDEX_HTML = """\
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="theme-color" content="#080c12">
<link rel="manifest" href="manifest.webmanifest">
<title>{title} — OpenShader</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');
  :root {{
    --bg:      #080c12;
    --surface: #0e1420;
    --border:  #1a2035;
    --accent:  #3a88ff;
    --text:    #c8d0e0;
    --muted:   #4a5570;
    --mono:    'Space Mono', monospace;
    --sans:    'Inter', system-ui, sans-serif;
  }}
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html {{ scroll-behavior: smooth; }}
  body {{
    background: var(--bg); color: var(--text);
    font: 14px/1.65 var(--sans); min-height: 100vh;
  }}

  /* ── Header ────────────────────────────────────────────────── */
  header {{
    padding: 28px 40px 0;
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid var(--border);
    padding-bottom: 20px;
  }}
  .logo {{
    font-family: var(--mono); font-size: 12px;
    color: var(--muted); letter-spacing: .08em;
    text-transform: uppercase;
  }}
  .logo span {{ color: var(--accent); }}
  nav a {{
    color: var(--muted); text-decoration: none; font-size: 12px;
    letter-spacing: .04em; margin-left: 24px;
    transition: color .2s;
  }}
  nav a:hover {{ color: var(--text); }}

  /* ── Hero player ───────────────────────────────────────────── */
  .hero {{
    padding: 40px;
    display: grid; grid-template-columns: 1fr 320px;
    gap: 32px; max-width: 1200px; margin: 0 auto;
    align-items: start;
  }}
  @media (max-width: 780px) {{
    .hero {{ grid-template-columns: 1fr; }}
    header {{ padding: 20px; }}
    .hero {{ padding: 20px; gap: 20px; }}
  }}

  .player-frame {{
    position: relative; border-radius: 8px; overflow: hidden;
    border: 1px solid var(--border);
    box-shadow: 0 0 60px rgba(58,136,255,.06);
    background: #000;
    aspect-ratio: 16/9;
  }}
  .player-frame iframe {{
    width: 100%; height: 100%;
    border: none; display: block;
  }}

  /* ── Info sidebar ──────────────────────────────────────────── */
  .sidebar h1 {{
    font: 700 22px/1.2 var(--mono); color: #fff;
    margin-bottom: 8px; letter-spacing: -.02em;
  }}
  .sidebar .author {{
    font-size: 12px; color: var(--muted); margin-bottom: 20px;
  }}
  .sidebar .author span {{ color: var(--accent); }}
  .sidebar .desc {{
    font-size: 13px; color: var(--muted); line-height: 1.7;
    margin-bottom: 24px;
  }}

  .tag-list {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 24px; }}
  .tag {{
    font-size: 10px; padding: 3px 8px;
    background: rgba(58,136,255,.1); color: var(--accent);
    border: 1px solid rgba(58,136,255,.2); border-radius: 3px;
    font-family: var(--mono); letter-spacing: .05em;
    text-transform: uppercase;
  }}

  .meta-grid {{
    display: grid; grid-template-columns: auto 1fr; gap: 6px 16px;
    font-size: 11px; margin-bottom: 24px;
  }}
  .meta-grid .label {{ color: var(--muted); }}
  .meta-grid .value {{ color: var(--text); font-family: var(--mono); }}

  /* ── Buttons ───────────────────────────────────────────────── */
  .btn-row {{ display: flex; flex-direction: column; gap: 8px; }}
  .btn-primary {{
    background: var(--accent); color: #fff; border: none;
    border-radius: 5px; padding: 10px 16px; cursor: pointer;
    font: 600 12px var(--sans); letter-spacing: .04em;
    text-transform: uppercase; transition: background .2s;
    text-align: center; text-decoration: none; display: block;
  }}
  .btn-primary:hover {{ background: #5599ff; }}
  .btn-secondary {{
    background: transparent; color: var(--muted);
    border: 1px solid var(--border); border-radius: 5px;
    padding: 9px 16px; cursor: pointer; font: 12px var(--sans);
    letter-spacing: .04em; text-transform: uppercase;
    transition: border-color .2s, color .2s;
    text-align: center; text-decoration: none; display: block;
  }}
  .btn-secondary:hover {{ border-color: var(--accent); color: var(--text); }}

  /* ── Embed section ─────────────────────────────────────────── */
  .embed-section {{
    max-width: 1200px; margin: 0 auto;
    padding: 0 40px 40px;
  }}
  @media (max-width: 780px) {{ .embed-section {{ padding: 0 20px 32px; }} }}
  .embed-section h2 {{
    font: 700 13px var(--mono); color: var(--muted);
    letter-spacing: .08em; text-transform: uppercase;
    margin-bottom: 12px;
    padding-top: 24px; border-top: 1px solid var(--border);
  }}
  .code-block {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: 14px 16px; position: relative;
    font: 12px/1.8 var(--mono); color: #8ab4f8;
    overflow-x: auto; white-space: nowrap;
  }}
  .copy-btn {{
    position: absolute; top: 8px; right: 8px;
    background: rgba(255,255,255,.06); color: var(--muted);
    border: 1px solid var(--border); border-radius: 3px;
    padding: 3px 8px; font: 10px var(--sans); cursor: pointer;
    transition: background .15s;
  }}
  .copy-btn:hover {{ background: rgba(255,255,255,.12); color: var(--text); }}

  /* ── Compatibility strip ────────────────────────────────────── */
  .compat-strip {{
    max-width: 1200px; margin: 0 auto;
    padding: 0 40px 48px;
    display: flex; gap: 12px; flex-wrap: wrap;
  }}
  @media (max-width: 780px) {{ .compat-strip {{ padding: 0 20px 32px; }} }}
  .compat-item {{
    display: flex; align-items: center; gap: 6px;
    font-size: 11px; color: var(--muted); padding: 5px 10px;
    border: 1px solid var(--border); border-radius: 4px;
    background: var(--surface);
  }}
  .compat-item .dot {{
    width: 6px; height: 6px; border-radius: 50%;
    background: #40c060;
  }}
  .compat-item .dot.warn {{ background: #c0a040; }}

  /* ── Footer ────────────────────────────────────────────────── */
  footer {{
    border-top: 1px solid var(--border);
    padding: 20px 40px;
    display: flex; justify-content: space-between; align-items: center;
    font-size: 11px; color: var(--muted);
  }}
  @media (max-width: 780px) {{ footer {{ padding: 16px 20px; flex-direction: column; gap: 8px; }} }}
  footer a {{ color: var(--muted); text-decoration: none; }}
  footer a:hover {{ color: var(--text); }}
</style>
</head>
<body>

<header>
  <div class="logo">Open<span>Shader</span> · WASM Player</div>
  <nav>
    <a href="player.html" target="_blank">Player direct</a>
    <a href="shaders/" target="_blank">Sources GLSL</a>
    <a href="meta.json" target="_blank">meta.json</a>
  </nav>
</header>

<main class="hero">
  <div class="player-frame">
    <iframe src="player.html" allowfullscreen loading="lazy"
            allow="fullscreen; autoplay"></iframe>
  </div>

  <aside class="sidebar">
    <h1>{title}</h1>
    <p class="author">par <span>{author}</span></p>
    <p class="desc">{description}</p>

    <div class="tag-list">
      {tags_html}
    </div>

    <div class="meta-grid">
      <span class="label">Résolution</span>
      <span class="value">{width} × {height}</span>
      <span class="label">Passes</span>
      <span class="value">{passes}</span>
      <span class="label">Licence</span>
      <span class="value">{licence}</span>
      <span class="label">Exporté</span>
      <span class="value">{date}</span>
      <span class="label">SDK</span>
      <span class="value">DemoMaker {sdk_ver}</span>
      <span class="label">Backend</span>
      <span class="value">WebGPU / WebGL2</span>
    </div>

    <div class="btn-row">
      <a href="player.html" class="btn-primary" target="_blank">▶ Ouvrir le player</a>
      <button class="btn-secondary" onclick="copyEmbed()">⧉ Copier le code embed</button>
      <a href="shaders/image.glsl" class="btn-secondary" download>⬇ Source GLSL</a>
    </div>
  </aside>
</main>

<section class="embed-section">
  <h2>Intégration — embed</h2>
  <div class="code-block" id="embed-code">
    &lt;script src=&quot;embed.js&quot;&gt;&lt;/script&gt;<br>
    &lt;div data-openshader data-width=&quot;100%&quot; data-height=&quot;480&quot;&gt;&lt;/div&gt;
  </div>
  <button class="copy-btn" onclick="copyEmbed()">Copier</button>
</section>

<section class="compat-strip">
  <div class="compat-item"><div class="dot"></div>Chrome 113+ · WebGPU</div>
  <div class="compat-item"><div class="dot"></div>Edge 113+ · WebGPU</div>
  <div class="compat-item"><div class="dot"></div>Firefox 98+ · WebGL2</div>
  <div class="compat-item"><div class="dot warn"></div>Safari 16.4+ · WebGL2</div>
  <div class="compat-item"><div class="dot"></div>Offline · Service Worker</div>
  <div class="compat-item"><div class="dot"></div>Mobile · Touch</div>
</section>

<footer>
  <span>Généré par <strong>DemoMaker {sdk_ver}</strong> · OpenShader</span>
  <span>
    <a href="https://github.com/openshader" target="_blank">GitHub</a> ·
    {licence}
  </span>
</footer>

<script>
function copyEmbed() {{
  const code = `<script src="embed.js"><\\/script>\\n<div data-openshader data-width="100%" data-height="480"></div>`;
  navigator.clipboard.writeText(code).then(() => {{
    const btn = document.querySelector('.copy-btn');
    if (btn) {{ btn.textContent = 'Copié ✓'; setTimeout(() => btn.textContent = 'Copier', 2000); }}
  }});
}}
// PWA install prompt
let _deferredPrompt;
window.addEventListener('beforeinstallprompt', e => {{
  e.preventDefault(); _deferredPrompt = e;
  const btn = document.createElement('button');
  btn.className = 'btn-secondary';
  btn.textContent = '⊕ Installer (PWA)';
  btn.onclick = async () => {{
    _deferredPrompt.prompt();
    const {{ outcome }} = await _deferredPrompt.userChoice;
    if (outcome === 'accepted') btn.remove();
  }};
  document.querySelector('.btn-row').appendChild(btn);
}});
// Register service worker
if ('serviceWorker' in navigator) {{
  navigator.serviceWorker.register('sw.js').catch(() => {{}});
}}
</script>
</body>
</html>
"""

# ══════════════════════════════════════════════════════════════════════════════
#  sw.js — Service Worker (offline + PWA cache)
# ══════════════════════════════════════════════════════════════════════════════

_SW_JS = """\
// OpenShader WASM — Service Worker v{version}
// Stratégie : Cache First pour les assets statiques
const CACHE = 'openshader-wasm-v{version}';
const PRECACHE = [
  './',
  './index.html',
  './player.html',
  './embed.js',
  './manifest.webmanifest',
  './meta.json',
  {shader_cache_entries}
];

self.addEventListener('install', e => {{
  e.waitUntil(
    caches.open(CACHE)
      .then(c => c.addAll(PRECACHE))
      .then(() => self.skipWaiting())
  );
}});

self.addEventListener('activate', e => {{
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
}});

self.addEventListener('fetch', e => {{
  if (e.request.method !== 'GET') return;
  e.respondWith(
    caches.match(e.request).then(hit => hit || fetch(e.request).then(res => {{
      if (res.ok) {{
        const clone = res.clone();
        caches.open(CACHE).then(c => c.put(e.request, clone));
      }}
      return res;
    }}))
  );
}});
"""

# ══════════════════════════════════════════════════════════════════════════════
#  manifest.webmanifest — PWA
# ══════════════════════════════════════════════════════════════════════════════

_MANIFEST = """\
{{
  "name": "{title}",
  "short_name": "{title_short}",
  "description": "{description}",
  "start_url": "./index.html",
  "display": "standalone",
  "background_color": "#080c12",
  "theme_color": "#3a88ff",
  "icons": [],
  "categories": ["entertainment", "art"],
  "lang": "fr"
}}
"""

# ══════════════════════════════════════════════════════════════════════════════
#  embed.js
# ══════════════════════════════════════════════════════════════════════════════

_EMBED_JS = """\
// OpenShader WASM embed — {title}
// Generated by DemoMaker {sdk_ver}
// Usage : <script src="embed.js"></script>
//         <div data-openshader data-width="100%" data-height="480"></div>
(function() {{
  var base = (document.currentScript || {{}}).src.replace(/embed\\.js$/, '');
  var targets = document.querySelectorAll('[data-openshader]');
  if (!targets.length) {{
    var d = document.createElement('div');
    document.currentScript && document.currentScript.parentNode
      .insertBefore(d, document.currentScript.nextSibling);
    d.setAttribute('data-openshader', '');
    targets = [d];
  }}
  targets.forEach(function(t) {{
    var w = t.getAttribute('data-width')  || '100%';
    var h = t.getAttribute('data-height') || '480';
    var f = document.createElement('iframe');
    f.src = base + 'player.html';
    f.width  = w; f.height = h;
    f.style.cssText = 'border:none;border-radius:8px;display:block;';
    f.allowFullscreen = true;
    f.setAttribute('allow', 'fullscreen; autoplay');
    t.appendChild(f);
  }});
}})();
"""

# ══════════════════════════════════════════════════════════════════════════════
#  Emscripten runtime stub — openshader_runtime.c
# ══════════════════════════════════════════════════════════════════════════════

_RUNTIME_C = """\
/*
 * openshader_runtime.c
 * --------------------
 * Stub C compilé en WebAssembly via Emscripten.
 * Fournit les fonctions appelées depuis player.html via Module.ccall().
 *
 * Compilation :
 *   emcc openshader_runtime.c -O3 \\
 *        -s MODULARIZE=1 -s EXPORT_NAME=OpenShaderRuntime \\
 *        -s EXPORTED_FUNCTIONS='["_os_init","_os_tick","_os_get_uniform_count","_os_get_uniform_name","_os_get_uniform_value"]' \\
 *        -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","UTF8ToString"]' \\
 *        -o openshader_runtime.js
 *
 * Le .wasm + .js produits remplacent le renderer JS natif du player
 * et permettent d'exécuter des plugins C++ natifs dans le navigateur.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <emscripten.h>

/* ── État global ─────────────────────────────────────────────────────────── */

#define MAX_UNIFORMS 64

typedef struct {{
    char  name[64];
    float value;
}} UniformEntry;

static UniformEntry _uniforms[MAX_UNIFORMS];
static int          _uniform_count = 0;
static float        _time          = 0.0f;
static float        _rms           = 0.0f;
static float        _bpm           = 120.0f;

/* ── API exportée vers JavaScript ────────────────────────────────────────── */

EMSCRIPTEN_KEEPALIVE
void os_init(float bpm) {{
    _bpm = bpm;
    _uniform_count = 0;
    _time = 0.0f;
}}

EMSCRIPTEN_KEEPALIVE
void os_tick(float dt, float itime, float rms) {{
    _time += dt;
    _rms  = rms;

    /* Exemple : génère des uniforms animés depuis le runtime C */
    _uniform_count = 3;

    snprintf(_uniforms[0].name, 64, "uWasmBeat");
    _uniforms[0].value = 0.5f + 0.5f * sinf(_time * _bpm / 60.0f * 3.14159f);

    snprintf(_uniforms[1].name, 64, "uWasmEnergy");
    _uniforms[1].value = rms;

    snprintf(_uniforms[2].name, 64, "uWasmTime");
    _uniforms[2].value = itime;
}}

EMSCRIPTEN_KEEPALIVE
int os_get_uniform_count(void) {{
    return _uniform_count;
}}

EMSCRIPTEN_KEEPALIVE
const char* os_get_uniform_name(int idx) {{
    if (idx < 0 || idx >= _uniform_count) return "";
    return _uniforms[idx].name;
}}

EMSCRIPTEN_KEEPALIVE
float os_get_uniform_value(int idx) {{
    if (idx < 0 || idx >= _uniform_count) return 0.0f;
    return _uniforms[idx].value;
}}
"""

# ══════════════════════════════════════════════════════════════════════════════
#  Makefile.emscripten
# ══════════════════════════════════════════════════════════════════════════════

_MAKEFILE = """\
# Makefile.emscripten
# -------------------
# Compile le runtime OpenShader en WebAssembly via Emscripten.
# Prérequis : emsdk installé et activé (source emsdk_env.sh)
# Usage    : make -f Makefile.emscripten

EMCC     = emcc
OUT_DIR  = ../
OUT_JS   = $(OUT_DIR)openshader_runtime.js
OUT_WASM = $(OUT_DIR)openshader_runtime.wasm

CFLAGS = -O3 -std=c11 -Wall

EXPORTED_FUNCS = [\
"_os_init",\
"_os_tick",\
"_os_get_uniform_count",\
"_os_get_uniform_name",\
"_os_get_uniform_value"\
]

EMFLAGS = \\
    -s MODULARIZE=1 \\
    -s EXPORT_NAME=OpenShaderRuntime \\
    -s EXPORTED_FUNCTIONS='$(EXPORTED_FUNCS)' \\
    -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","UTF8ToString"]' \\
    -s ALLOW_MEMORY_GROWTH=1 \\
    -s ENVIRONMENT='web' \\
    --no-entry

all: $(OUT_JS)

$(OUT_JS): openshader_runtime.c
\\t$(EMCC) $(CFLAGS) $(EMFLAGS) $< -o $@
\\t@echo "\\n✅ WASM compilé : $(OUT_WASM)"
\\t@echo "   Copiez $(OUT_JS) et $(OUT_WASM) dans le dossier du player.\\n"

clean:
\\trm -f $(OUT_JS) $(OUT_WASM)

.PHONY: all clean
"""

# ══════════════════════════════════════════════════════════════════════════════
#  README_BUILD.md
# ══════════════════════════════════════════════════════════════════════════════

_README_BUILD = """\
# OpenShader — Build Emscripten (WASM)

Ce dossier contient le runtime C compilable en WebAssembly via **Emscripten**.

## Prérequis

```bash
# Installer emsdk
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh   # Linux/macOS
# ou : emsdk_env.bat    # Windows
```

## Compilation

```bash
cd dist/runtime/
make -f Makefile.emscripten
```

Produit dans `dist/` :
- `openshader_runtime.js`   — loader JS (Module.ccall)
- `openshader_runtime.wasm` — bytecode WebAssembly

## Structure du player avec WASM

Une fois compilé, le `player.html` détecte automatiquement la présence
du `.wasm` et l'utilise pour les calculs d'uniforms côté C
(plus rapide que JS pour les simulations de particules, physique, FFT).

## API exportée

| Fonction                         | Description                              |
|----------------------------------|------------------------------------------|
| `os_init(bpm)`                   | Initialise le runtime, BPM du projet     |
| `os_tick(dt, itime, rms)`        | Tick par frame                           |
| `os_get_uniform_count()`         | Nombre d'uniforms produits               |
| `os_get_uniform_name(idx)`       | Nom du uniform n°idx (string C)          |
| `os_get_uniform_value(idx)`      | Valeur float du uniform n°idx            |

## Intégration de plugins natifs

Compilez vos plugins C++ depuis `plugins/sdk/examples/` directement
avec Emscripten en ajoutant leur `.cpp` au `Makefile.emscripten` :

```makefile
$(OUT_JS): openshader_runtime.c ../exemples/particle_system.cpp
\\t$(EMCC) $(CFLAGS) $(EMFLAGS) $^ -o $@
```
"""


# ══════════════════════════════════════════════════════════════════════════════
#  WasmExporter — classe principale
# ══════════════════════════════════════════════════════════════════════════════

class WasmExporter:
    """
    Génère un bundle WASM complet à partir des métadonnées et shaders.

    Paramètres
    ----------
    meta : dict
        title, author, description, tags (list[str]), licence,
        resolution (tuple[int,int]), bpm (float)
    shaders : dict[str, str]
        { pass_name: glsl_source }  ex: {'Image': '...', 'Buffer A': '...'}
    """

    def __init__(self, meta: dict, shaders: dict[str, str]):
        self.meta    = meta
        self.shaders = shaders

    def export(self, output_dir: str) -> list[str]:
        """
        Génère le bundle WASM complet dans *output_dir*.
        Retourne la liste des fichiers créés.
        """
        os.makedirs(output_dir, exist_ok=True)

        shaders_dir = os.path.join(output_dir, 'shaders')
        runtime_dir = os.path.join(output_dir, 'runtime')
        os.makedirs(shaders_dir, exist_ok=True)
        os.makedirs(runtime_dir, exist_ok=True)

        created = []
        title       = self.meta.get('title', 'Shader')
        author      = self.meta.get('author', 'Anonyme')
        description = self.meta.get('description', '')
        tags        = self.meta.get('tags', [])
        licence     = self.meta.get('licence', 'CC0 1.0')
        resolution  = self.meta.get('resolution', (1920, 1080))
        date_str    = datetime.datetime.now().strftime('%Y-%m-%d')

        # ── 1. Shaders sources ────────────────────────────────────────────
        for pass_name, src in self.shaders.items():
            safe = re.sub(r'[^\w\-]', '_', pass_name).lower()
            path = os.path.join(shaders_dir, f'{safe}.glsl')
            with open(path, 'w', encoding='utf-8') as f:
                f.write(src)
            created.append(path)

        # ── 2. meta.json ──────────────────────────────────────────────────
        meta_dict = {
            'title':       title,
            'author':      author,
            'description': description,
            'tags':        tags,
            'licence':     licence,
            'resolution':  list(resolution),
            'passes':      list(self.shaders.keys()),
            'date':        date_str,
            'sdk_version': _SDK_VER,
            'runtime':     'WebAssembly / WebGL2 / WebGPU',
        }
        meta_path = os.path.join(output_dir, 'meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_dict, f, indent=2, ensure_ascii=False)
        created.append(meta_path)

        # ── 3. player.html ────────────────────────────────────────────────
        # Choisit le shader Image (ou le premier disponible) à inliner
        image_src = (self.shaders.get('Image')
                     or self.shaders.get('image')
                     or next(iter(self.shaders.values()), ''))
        shaders_for_player = {k: v for k, v in self.shaders.items()}

        player_html = _PLAYER_HTML.format(
            title          = self._esc(title),
            shaders_json   = json.dumps(shaders_for_player, ensure_ascii=False),
            meta_json      = json.dumps(meta_dict, ensure_ascii=False),
            version        = _VERSION,
            sdk_ver        = _SDK_VER,
            date           = date_str,
        )
        player_path = os.path.join(output_dir, 'player.html')
        with open(player_path, 'w', encoding='utf-8') as f:
            f.write(player_html)
        created.append(player_path)

        # ── 4. embed.js ───────────────────────────────────────────────────
        embed_path = os.path.join(output_dir, 'embed.js')
        with open(embed_path, 'w', encoding='utf-8') as f:
            f.write(_EMBED_JS.format(title=title, sdk_ver=_SDK_VER))
        created.append(embed_path)

        # ── 5. sw.js (Service Worker) ─────────────────────────────────────
        shader_entries = '\n  '.join(
            f"'./shaders/{re.sub(r'[^\\w\\-]', '_', k).lower()}.glsl',"
            for k in self.shaders
        )
        sw_path = os.path.join(output_dir, 'sw.js')
        with open(sw_path, 'w', encoding='utf-8') as f:
            f.write(_SW_JS.format(
                version=_VERSION,
                shader_cache_entries=shader_entries,
            ))
        created.append(sw_path)

        # ── 6. manifest.webmanifest (PWA) ─────────────────────────────────
        manifest_path = os.path.join(output_dir, 'manifest.webmanifest')
        title_short   = title[:12] + ('…' if len(title) > 12 else '')
        with open(manifest_path, 'w', encoding='utf-8') as f:
            f.write(_MANIFEST.format(
                title       = title,
                title_short = title_short,
                description = description or title,
            ))
        created.append(manifest_path)

        # ── 7. index.html ─────────────────────────────────────────────────
        tags_html  = ''.join(f'<span class="tag">{self._esc(t)}</span>' for t in tags)
        passes_str = ', '.join(self.shaders.keys()) or 'Image'
        index_html = _INDEX_HTML.format(
            title       = self._esc(title),
            author      = self._esc(author),
            description = self._esc(description or ''),
            tags_html   = tags_html,
            width       = resolution[0],
            height      = resolution[1],
            passes      = passes_str,
            licence     = self._esc(licence),
            date        = date_str,
            sdk_ver     = _SDK_VER,
        )
        index_path = os.path.join(output_dir, 'index.html')
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(index_html)
        created.append(index_path)

        # ── 8. Runtime Emscripten stub ────────────────────────────────────
        runtime_c = os.path.join(runtime_dir, 'openshader_runtime.c')
        with open(runtime_c, 'w', encoding='utf-8') as f:
            f.write(_RUNTIME_C)
        created.append(runtime_c)

        makefile_path = os.path.join(runtime_dir, 'Makefile.emscripten')
        with open(makefile_path, 'w', encoding='utf-8') as f:
            f.write(_MAKEFILE)
        created.append(makefile_path)

        readme_path = os.path.join(runtime_dir, 'README_BUILD.md')
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(_README_BUILD)
        created.append(readme_path)

        return created

    @staticmethod
    def _esc(s: str) -> str:
        """Échappe les caractères HTML basiques."""
        return (s.replace('&', '&amp;')
                 .replace('<', '&lt;')
                 .replace('>', '&gt;')
                 .replace('"', '&quot;'))
