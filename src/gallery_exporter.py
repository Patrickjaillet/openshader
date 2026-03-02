"""
gallery_exporter.py
-------------------
v2.7 — Export vers galerie en ligne.

Fonctionnalités :
  - GalleryExporter  : génère un dossier galerie complet
      · index.html        — page vitrine avec embed player WebGL2
      · player.html       — iframe standalone embed (WebGL2, zéro dépendance)
      · embed.js          — snippet <script> pour intégration externe
      · meta.json         — métadonnées (titre, auteur, tags, licence, date)
      · preview.png       — capture du viewport (si disponible)
      · shaders/          — sources GLSL du projet
  - GalleryPublishDialog : UI de publication (titre, auteur, tags, licence,
      description, preview, dossier de sortie)

Le player HTML5/WebGL2 est auto-contenu (pas de CDN, pas de serveur) :
  - Compile et exécute le shader Image/mainImage directement dans un <canvas>
  - Fournit les uniforms Shadertoy standard (iTime, iResolution, iMouse…)
  - Boutons Play/Pause, Fullscreen, info overlay
  - Responsive : s'adapte au conteneur parent (idéal iframe)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import datetime
from pathlib import Path

from PyQt6.QtCore    import Qt, QSettings
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QTextEdit, QPushButton, QComboBox,
    QFileDialog, QMessageBox, QCheckBox, QFrame, QWidget,
    QSizePolicy, QScrollArea,
)
from PyQt6.QtGui import QFont, QPixmap

from .logger import get_logger

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Licences disponibles
# ══════════════════════════════════════════════════════════════════════════════

LICENCES = [
    "CC0 1.0 — Domaine public",
    "CC BY 4.0 — Attribution",
    "CC BY-SA 4.0 — Attribution + Partage à l'identique",
    "CC BY-NC 4.0 — Attribution + Non commercial",
    "CC BY-NC-SA 4.0 — Attribution + Non commercial + Partage",
    "MIT",
    "Tous droits réservés",
]

LICENCE_URLS = {
    "CC0 1.0 — Domaine public":              "https://creativecommons.org/publicdomain/zero/1.0/",
    "CC BY 4.0 — Attribution":               "https://creativecommons.org/licenses/by/4.0/",
    "CC BY-SA 4.0 — Attribution + Partage à l'identique": "https://creativecommons.org/licenses/by-sa/4.0/",
    "CC BY-NC 4.0 — Attribution + Non commercial":        "https://creativecommons.org/licenses/by-nc/4.0/",
    "CC BY-NC-SA 4.0 — Attribution + Non commercial + Partage": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "MIT":                                   "https://opensource.org/licenses/MIT",
    "Tous droits réservés":                  "",
}


# ══════════════════════════════════════════════════════════════════════════════
#  Template HTML5 — index.html (page vitrine)
# ══════════════════════════════════════════════════════════════════════════════

_INDEX_HTML = """\
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — OpenShader Gallery</title>
<meta name="description" content="{description_short}">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{description_short}">
<meta property="og:image" content="preview.png">
<meta property="og:type" content="website">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0a0c10; color: #c0c4d0; min-height: 100vh;
  }}
  .header {{
    background: #12141a; border-bottom: 1px solid #1e2030;
    padding: 16px 32px; display: flex; align-items: center; gap: 16px;
  }}
  .header-logo {{
    font-size: 20px; font-weight: 700; color: #3a88ff; letter-spacing: -0.5px;
  }}
  .header-subtitle {{ font-size: 12px; color: #6a7090; }}
  .main {{ max-width: 1100px; margin: 0 auto; padding: 32px 24px; }}
  .player-wrap {{
    background: #000; border-radius: 8px; overflow: hidden;
    box-shadow: 0 8px 40px rgba(0,0,0,.7);
    position: relative; aspect-ratio: 16/9;
  }}
  .player-wrap iframe {{
    width: 100%; height: 100%; border: none; display: block;
  }}
  .meta-section {{
    margin-top: 28px; display: grid;
    grid-template-columns: 1fr auto; gap: 24px; align-items: start;
  }}
  h1 {{ font-size: 26px; font-weight: 700; color: #e0e4f0; margin-bottom: 6px; }}
  .author {{ color: #6a90c8; font-size: 14px; margin-bottom: 12px; }}
  .description {{ font-size: 14px; line-height: 1.65; color: #a0a8c0; max-width: 680px; }}
  .tags {{ margin-top: 14px; display: flex; flex-wrap: wrap; gap: 6px; }}
  .tag {{
    background: #1a2040; color: #6080c0; border: 1px solid #2a3060;
    border-radius: 20px; padding: 3px 12px; font-size: 12px;
  }}
  .side-panel {{ min-width: 200px; }}
  .info-card {{
    background: #12141a; border: 1px solid #1e2030; border-radius: 6px;
    padding: 16px; font-size: 13px;
  }}
  .info-card dt {{ color: #6a7090; font-size: 11px; text-transform: uppercase;
    letter-spacing: .5px; margin-top: 12px; }}
  .info-card dt:first-child {{ margin-top: 0; }}
  .info-card dd {{ color: #c0c4d0; margin-top: 3px; }}
  .info-card a {{ color: #3a88ff; text-decoration: none; }}
  .embed-section {{
    margin-top: 32px; background: #0e1018; border: 1px solid #1e2030;
    border-radius: 6px; padding: 20px;
  }}
  .embed-section h3 {{ font-size: 14px; color: #a0a8c0; margin-bottom: 12px; }}
  .embed-code {{
    background: #080a10; border: 1px solid #1e2030; border-radius: 4px;
    padding: 12px 16px; font-family: 'Consolas', monospace; font-size: 12px;
    color: #80c0a0; white-space: pre; overflow-x: auto;
  }}
  .btn-copy {{
    margin-top: 8px; background: #1e2030; color: #8090b0;
    border: 1px solid #2a2d3a; border-radius: 3px; padding: 5px 14px;
    font-size: 12px; cursor: pointer;
  }}
  .btn-copy:hover {{ background: #2a3050; color: #c0c8e0; }}
  .footer {{
    margin-top: 48px; border-top: 1px solid #1e2030; padding: 16px 24px;
    text-align: center; font-size: 11px; color: #404860;
  }}
  @media (max-width: 700px) {{
    .meta-section {{ grid-template-columns: 1fr; }}
    .side-panel {{ order: -1; }}
  }}
</style>
</head>
<body>
<header class="header">
  <div>
    <div class="header-logo">⬡ OpenShader</div>
    <div class="header-subtitle">Gallery</div>
  </div>
</header>

<main class="main">
  <div class="player-wrap">
    <iframe src="player.html" allowfullscreen></iframe>
  </div>

  <div class="meta-section">
    <div>
      <h1>{title}</h1>
      <div class="author">par <strong>{author}</strong></div>
      <p class="description">{description}</p>
      <div class="tags">
        {tags_html}
      </div>
    </div>
    <aside class="side-panel">
      <dl class="info-card">
        <dt>Date</dt>
        <dd>{date}</dd>
        <dt>Résolution</dt>
        <dd>{resolution}</dd>
        <dt>Licence</dt>
        <dd>{licence_html}</dd>
        {shader_count_html}
      </dl>
    </aside>
  </div>

  <div class="embed-section">
    <h3>🔗 Intégrer sur votre site</h3>
    <div class="embed-code" id="embed-code">&lt;iframe
  src="{embed_url}"
  width="960" height="540"
  frameborder="0" allowfullscreen
  style="border-radius:6px;"&gt;
&lt;/iframe&gt;</div>
    <button class="btn-copy" onclick="copyEmbed()">📋 Copier le code</button>
    <p style="margin-top:8px;font-size:11px;color:#404860;">
      Ou via script : <code style="color:#80c0a0;">&lt;script src="embed.js"&gt;&lt;/script&gt;</code>
    </p>
  </div>
</main>

<footer class="footer">
  Généré avec <strong>OpenShader / OpenShader v2.7</strong> —
  <a href="player.html" style="color:#3a88ff;">Ouvrir le player</a>
</footer>

<script>
function copyEmbed() {{
  const txt = document.getElementById('embed-code').textContent;
  navigator.clipboard.writeText(txt).then(() => {{
    const btn = document.querySelector('.btn-copy');
    btn.textContent = '✓ Copié !';
    setTimeout(() => btn.textContent = '📋 Copier le code', 2000);
  }});
}}
</script>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  Template HTML5 — player.html (WebGL2, zéro dépendance)
# ══════════════════════════════════════════════════════════════════════════════

_PLAYER_HTML = """\
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{
    width: 100%; height: 100%; background: #000; overflow: hidden;
  }}
  canvas {{
    display: block; width: 100%; height: 100%;
    image-rendering: pixelated;
  }}
  #ui {{
    position: absolute; bottom: 0; left: 0; right: 0;
    padding: 10px 14px; background: linear-gradient(transparent, rgba(0,0,0,.8));
    display: flex; align-items: center; gap: 10px;
    opacity: 0; transition: opacity .3s; pointer-events: none;
  }}
  body:hover #ui {{ opacity: 1; pointer-events: auto; }}
  .btn {{
    background: rgba(255,255,255,.12); color: #fff; border: none;
    border-radius: 4px; padding: 5px 10px; cursor: pointer; font-size: 13px;
    backdrop-filter: blur(4px);
  }}
  .btn:hover {{ background: rgba(255,255,255,.25); }}
  #time-display {{ color: rgba(255,255,255,.7); font-size: 12px;
    font-family: monospace; margin-left: auto; }}
  #title-overlay {{
    position: absolute; top: 12px; left: 14px;
    color: rgba(255,255,255,.5); font-family: 'Segoe UI', sans-serif;
    font-size: 12px; pointer-events: none;
  }}
  #error-overlay {{
    display: none; position: absolute; inset: 0;
    background: rgba(0,0,0,.92); color: #ff6060;
    font-family: monospace; font-size: 12px;
    padding: 24px; white-space: pre-wrap; overflow: auto;
  }}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="title-overlay">{title} <span style="color:#3a88ff;">▶</span></div>
<div id="error-overlay"></div>
<div id="ui">
  <button class="btn" id="btn-play" onclick="togglePlay()">⏸</button>
  <button class="btn" onclick="restart()">⟳</button>
  <span id="time-display">0.00s</span>
  <button class="btn" onclick="toggleFullscreen()" style="margin-left:auto;">⛶</button>
</div>

<script>
// ── Shader source ─────────────────────────────────────────────────────────────
const GLSL_SOURCE = {glsl_source_json};

// ── WebGL2 setup ──────────────────────────────────────────────────────────────
const canvas = document.getElementById('c');
const gl = canvas.getContext('webgl2', {{antialias: false, preserveDrawingBuffer: false}});
const errDiv = document.getElementById('error-overlay');

if (!gl) {{
  errDiv.style.display = 'block';
  errDiv.textContent = '❌ WebGL2 non supporté par ce navigateur.\\nEssayez Chrome ou Firefox à jour.';
}}

const VERT = `#version 300 es
in vec2 a_pos;
void main() {{ gl_Position = vec4(a_pos, 0.0, 1.0); }}`;

// Adapt Shadertoy mainImage → WebGL2 main
function adaptShader(src) {{
  // Retire l'éventuel #version existant
  src = src.replace(/#version\\s+\\d+\\s*(core|compatibility)?\\s*\\n/g, '');
  // Retire les déclarations uniform Shadertoy déjà présentes
  src = src.replace(/^\\s*uniform\\s+(vec[234]|float|int|sampler2D)\\s+i[A-Z]\\w*\\s*;/gm, '');
  src = src.replace(/^\\s*uniform\\s+(vec[234]|float|int|sampler2D)\\s+u[A-Z]\\w*\\s*;/gm, '');

  const header = `#version 300 es
precision highp float;
precision highp int;
uniform vec3  iResolution;
uniform float iTime;
uniform float iTimeDelta;
uniform int   iFrame;
uniform vec4  iMouse;
uniform float iSampleRate;
out vec4 _fragOut;
`;

  // mainImage → appel depuis main()
  const hasMainImage = /void\\s+mainImage\\s*\\(/.test(src);
  const footer = hasMainImage
    ? `\\nvoid main() {{ mainImage(_fragOut, gl_FragCoord.xy); }}`
    : `\\nvoid main() {{ _fragOut = vec4(0.); }}`;

  return header + src + footer;
}}

function compileShader(type, source) {{
  const s = gl.createShader(type);
  gl.shaderSource(s, source);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {{
    throw new Error(gl.getShaderInfoLog(s));
  }}
  return s;
}}

let prog = null;
function buildProgram() {{
  const fragSrc = adaptShader(GLSL_SOURCE);
  try {{
    const vert = compileShader(gl.VERTEX_SHADER, VERT);
    const frag = compileShader(gl.FRAGMENT_SHADER, fragSrc);
    const p = gl.createProgram();
    gl.attachShader(p, vert); gl.attachShader(p, frag);
    gl.bindAttribLocation(p, 0, 'a_pos');
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {{
      throw new Error(gl.getProgramInfoLog(p));
    }}
    if (prog) gl.deleteProgram(prog);
    prog = p;
    errDiv.style.display = 'none';
  }} catch(e) {{
    errDiv.style.display = 'block';
    errDiv.textContent = '❌ Erreur de compilation GLSL :\\n' + e.message;
  }}
}}

// Quad plein écran
const vao = gl.createVertexArray();
gl.bindVertexArray(vao);
const buf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, buf);
gl.bufferData(gl.ARRAY_BUFFER,
  new Float32Array([-1,-1, 1,-1, -1,1, 1,-1, 1,1, -1,1]), gl.STATIC_DRAW);
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
gl.bindVertexArray(null);

buildProgram();

// ── Uniforms ──────────────────────────────────────────────────────────────────
function getUnif(name) {{
  return prog ? gl.getUniformLocation(prog, name) : null;
}}

// ── Boucle de rendu ───────────────────────────────────────────────────────────
let startTime = performance.now() / 1000;
let pausedAt  = null;
let pausedOffset = 0;
let frame = 0;
let lastTime = 0;
let mouse = [0, 0, 0, 0];
let playing = true;

function currentTime() {{
  if (!playing) return pausedOffset;
  return (performance.now() / 1000 - startTime) + pausedOffset;
}}

function resize() {{
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(canvas.clientWidth  * dpr);
  const h = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {{
    canvas.width = w; canvas.height = h;
  }}
}}

function render(ts) {{
  if (!prog) {{ requestAnimationFrame(render); return; }}
  resize();
  const t = currentTime();
  const dt = Math.max(0, t - lastTime);
  lastTime = t;

  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.useProgram(prog);
  gl.bindVertexArray(vao);

  gl.uniform3f(getUnif('iResolution'), canvas.width, canvas.height, 1.0);
  gl.uniform1f(getUnif('iTime'),       t);
  gl.uniform1f(getUnif('iTimeDelta'),  dt);
  gl.uniform1i(getUnif('iFrame'),      frame);
  gl.uniform4fv(getUnif('iMouse'),     mouse);
  gl.uniform1f(getUnif('iSampleRate'), 44100.0);

  gl.drawArrays(gl.TRIANGLES, 0, 6);
  gl.bindVertexArray(null);

  frame++;
  document.getElementById('time-display').textContent = t.toFixed(2) + 's';
  if (playing) requestAnimationFrame(render);
}}

requestAnimationFrame(render);

// ── Mouse ─────────────────────────────────────────────────────────────────────
canvas.addEventListener('mousemove', e => {{
  const r = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  mouse[0] = (e.clientX - r.left) * dpr;
  mouse[1] = canvas.height - (e.clientY - r.top) * dpr;
}});
canvas.addEventListener('mousedown', e => {{
  const r = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  mouse[2] = (e.clientX - r.left) * dpr;
  mouse[3] = canvas.height - (e.clientY - r.top) * dpr;
}});
canvas.addEventListener('mouseup', () => {{ mouse[2] = 0; mouse[3] = 0; }});

// ── Controls ──────────────────────────────────────────────────────────────────
function togglePlay() {{
  playing = !playing;
  document.getElementById('btn-play').textContent = playing ? '⏸' : '▶';
  if (playing) {{
    startTime = performance.now() / 1000;
    requestAnimationFrame(render);
  }} else {{
    pausedOffset = currentTime();
  }}
}}
function restart() {{
  startTime = performance.now() / 1000;
  pausedOffset = 0;
  frame = 0;
  lastTime = 0;
  if (!playing) {{ playing = true; document.getElementById('btn-play').textContent = '⏸'; requestAnimationFrame(render); }}
}}
function toggleFullscreen() {{
  if (!document.fullscreenElement) canvas.requestFullscreen();
  else document.exitFullscreen();
}}
document.addEventListener('keydown', e => {{
  if (e.code === 'Space') {{ e.preventDefault(); togglePlay(); }}
  if (e.code === 'Escape') {{ if (document.fullscreenElement) document.exitFullscreen(); }}
}});
</script>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  Template embed.js
# ══════════════════════════════════════════════════════════════════════════════

_EMBED_JS = """\
// OpenShader embed player — {title}
// Généré par OpenShader v2.7
// Usage : <script src="embed.js"></script>
// Insère automatiquement un iframe 16:9 dans l'élément portant data-openshader
(function() {{
  var base = (document.currentScript || {{}}).src.replace(/embed\\.js$/, '');
  var targets = document.querySelectorAll('[data-openshader]');
  if (!targets.length) {{
    // Fallback : insère après la balise script
    var d = document.createElement('div');
    d.setAttribute('data-openshader', '');
    document.currentScript && document.currentScript.parentNode.insertBefore(d, document.currentScript.nextSibling);
    targets = [d];
  }}
  targets.forEach(function(t) {{
    var w = t.getAttribute('data-width')  || '100%';
    var h = t.getAttribute('data-height') || '540';
    var iframe = document.createElement('iframe');
    iframe.src = base + 'player.html';
    iframe.width  = w;
    iframe.height = h;
    iframe.style.cssText = 'border:none;border-radius:6px;display:block;';
    iframe.allowFullscreen = true;
    t.appendChild(iframe);
  }});
}})();
"""


# ══════════════════════════════════════════════════════════════════════════════
#  GalleryExporter
# ══════════════════════════════════════════════════════════════════════════════

class GalleryExporter:
    """
    Génère un dossier galerie complet à partir des métadonnées et des shaders.

    Paramètres
    ----------
    meta : dict
        title, author, description, tags (list[str]), licence,
        resolution (tuple[int,int])
    shaders : dict[str, str]
        { pass_name: glsl_source }  (ex: {'Image': '...', 'Buffer A': '...'})
    preview_pixmap : QPixmap | None
        Capture du viewport (optionnel, sauvée en PNG)
    """

    def __init__(self,
                 meta: dict,
                 shaders: dict[str, str],
                 preview_pixmap=None):
        self.meta            = meta
        self.shaders         = shaders
        self.preview_pixmap  = preview_pixmap

    # ── Public API ────────────────────────────────────────────────────────────

    def export(self, output_dir: str) -> list[str]:
        """
        Génère tous les fichiers dans *output_dir*.
        Retourne la liste des fichiers créés.
        """
        os.makedirs(output_dir, exist_ok=True)
        shaders_dir = os.path.join(output_dir, 'shaders')
        os.makedirs(shaders_dir, exist_ok=True)

        created = []

        # 1. Shaders sources
        for pass_name, src in self.shaders.items():
            safe = re.sub(r'[^\w\-]', '_', pass_name).lower()
            path = os.path.join(shaders_dir, f'{safe}.glsl')
            with open(path, 'w', encoding='utf-8') as f:
                f.write(src)
            created.append(path)

        # 2. meta.json
        meta_path = os.path.join(output_dir, 'meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(self._build_meta_dict(), f, indent=2, ensure_ascii=False)
        created.append(meta_path)

        # 3. player.html
        player_path = os.path.join(output_dir, 'player.html')
        with open(player_path, 'w', encoding='utf-8') as f:
            f.write(self._build_player_html())
        created.append(player_path)

        # 4. embed.js
        embed_js_path = os.path.join(output_dir, 'embed.js')
        with open(embed_js_path, 'w', encoding='utf-8') as f:
            f.write(_EMBED_JS.format(title=self._safe_html(self.meta.get('title', 'Shader'))))
        created.append(embed_js_path)

        # 5. index.html
        index_path = os.path.join(output_dir, 'index.html')
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(self._build_index_html())
        created.append(index_path)

        # 6. preview.png (optionnel)
        if self.preview_pixmap and not self.preview_pixmap.isNull():
            preview_path = os.path.join(output_dir, 'preview.png')
            self.preview_pixmap.save(preview_path, 'PNG')
            created.append(preview_path)
            log.info("Preview PNG sauvée : %s", preview_path)

        log.info("Galerie exportée : %d fichiers dans %s", len(created), output_dir)
        return created

    # ── Builders privés ───────────────────────────────────────────────────────

    def _build_meta_dict(self) -> dict:
        m = self.meta
        w, h = m.get('resolution', (1920, 1080))
        return {
            "title":       m.get('title', 'Untitled'),
            "author":      m.get('author', ''),
            "description": m.get('description', ''),
            "tags":        m.get('tags', []),
            "licence":     m.get('licence', 'Tous droits réservés'),
            "licence_url": LICENCE_URLS.get(m.get('licence', ''), ''),
            "resolution":  f"{w}x{h}",
            "date":        datetime.date.today().isoformat(),
            "generator":   "OpenShader v2.7 / OpenShader",
            "passes":      list(self.shaders.keys()),
        }

    def _build_player_html(self) -> str:
        # Sélectionne la passe Image en priorité, sinon la première
        src = (self.shaders.get('Image')
               or self.shaders.get('image')
               or next(iter(self.shaders.values()), ''))
        glsl_json = json.dumps(src)
        return _PLAYER_HTML.format(
            title=self._safe_html(self.meta.get('title', 'Shader')),
            glsl_source_json=glsl_json,
        )

    def _build_index_html(self) -> str:
        m        = self.meta
        title    = self._safe_html(m.get('title', 'Untitled'))
        author   = self._safe_html(m.get('author', ''))
        desc     = self._safe_html(m.get('description', ''))
        desc_short = desc[:160] + ('…' if len(desc) > 160 else '')
        tags_html = ''.join(
            f'<span class="tag">{self._safe_html(t)}</span>'
            for t in m.get('tags', [])
        )
        w, h = m.get('resolution', (1920, 1080))

        licence     = m.get('licence', 'Tous droits réservés')
        licence_url = LICENCE_URLS.get(licence, '')
        if licence_url:
            licence_html = f'<a href="{licence_url}" target="_blank" rel="noopener">{self._safe_html(licence)}</a>'
        else:
            licence_html = self._safe_html(licence)

        n = len(self.shaders)
        shader_count_html = (
            f'<dt>Passes</dt><dd>{n} passe{"s" if n > 1 else ""}</dd>'
            if n > 1 else ''
        )

        return _INDEX_HTML.format(
            title=title,
            author=author,
            description=desc,
            description_short=desc_short,
            tags_html=tags_html,
            date=datetime.date.today().strftime('%d %B %Y'),
            resolution=f'{w} × {h}',
            licence_html=licence_html,
            shader_count_html=shader_count_html,
            embed_url='player.html',
        )

    @staticmethod
    def _safe_html(s: str) -> str:
        return (str(s)
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;'))


# ══════════════════════════════════════════════════════════════════════════════
#  GalleryPublishDialog
# ══════════════════════════════════════════════════════════════════════════════

class GalleryPublishDialog(QDialog):
    """
    Dialog de publication galerie.

    Paramètres
    ----------
    shaders       : dict[str, str]  — passes GLSL du projet
    preview_pixmap: QPixmap | None  — capture du viewport
    parent        : QWidget | None
    """

    def __init__(self,
                 shaders: dict[str, str],
                 preview_pixmap=None,
                 resolution: tuple[int, int] = (1920, 1080),
                 parent=None):
        super().__init__(parent)
        self.shaders         = shaders
        self.preview_pixmap  = preview_pixmap
        self.resolution      = resolution
        self._output_dir: str | None = None

        self.setWindowTitle("🌐 Publier vers la galerie en ligne")
        self.setMinimumSize(640, 620)
        self.resize(700, 680)
        self._build_ui()
        self._load_settings()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        vl = QVBoxLayout(content)
        vl.setContentsMargins(24, 20, 24, 16)
        vl.setSpacing(16)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        # ── Section Métadonnées ──────────────────────────────────────────────
        grp_meta = QGroupBox("Métadonnées")
        form = QFormLayout(grp_meta)
        form.setSpacing(10)
        form.setContentsMargins(16, 16, 16, 16)

        self._edit_title = QLineEdit()
        self._edit_title.setPlaceholderText("Nom de votre création…")
        form.addRow("Titre *", self._edit_title)

        self._edit_author = QLineEdit()
        self._edit_author.setPlaceholderText("Votre pseudo ou nom…")
        form.addRow("Auteur *", self._edit_author)

        self._edit_desc = QTextEdit()
        self._edit_desc.setPlaceholderText(
            "Description de votre shader (technique, inspiration, contexte…)")
        self._edit_desc.setFixedHeight(90)
        form.addRow("Description", self._edit_desc)

        self._edit_tags = QLineEdit()
        self._edit_tags.setPlaceholderText("glsl, raymarching, fractal, audio-reactive… (séparés par des virgules)")
        form.addRow("Tags", self._edit_tags)

        self._cmb_licence = QComboBox()
        self._cmb_licence.addItems(LICENCES)
        form.addRow("Licence", self._cmb_licence)

        vl.addWidget(grp_meta)

        # ── Section Preview ──────────────────────────────────────────────────
        grp_prev = QGroupBox("Preview")
        hl_prev = QHBoxLayout(grp_prev)
        hl_prev.setContentsMargins(16, 12, 16, 12)
        hl_prev.setSpacing(14)

        self._lbl_preview = QLabel()
        self._lbl_preview.setFixedSize(192, 108)
        self._lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_preview.setStyleSheet(
            "border: 1px solid palette(mid); border-radius: 4px; background: #000;")
        if self.preview_pixmap and not self.preview_pixmap.isNull():
            self._lbl_preview.setPixmap(
                self.preview_pixmap.scaled(192, 108,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
        else:
            self._lbl_preview.setText("Aucune\npreview")
        hl_prev.addWidget(self._lbl_preview)

        prev_info = QVBoxLayout()
        prev_info.setSpacing(6)
        prev_info.addWidget(QLabel(
            "La preview est une capture du viewport actuel.\n"
            "Elle sera affichée dans la page galerie et les\n"
            "métadonnées Open Graph (partage réseaux sociaux)."))
        self._chk_include_preview = QCheckBox("Inclure la preview dans l'export")
        self._chk_include_preview.setChecked(
            self.preview_pixmap is not None and not self.preview_pixmap.isNull())
        self._chk_include_preview.setEnabled(
            self.preview_pixmap is not None and not self.preview_pixmap.isNull())
        prev_info.addWidget(self._chk_include_preview)
        prev_info.addStretch()
        hl_prev.addLayout(prev_info, 1)

        vl.addWidget(grp_prev)

        # ── Section Contenu ──────────────────────────────────────────────────
        grp_content = QGroupBox("Contenu exporté")
        fc = QVBoxLayout(grp_content)
        fc.setContentsMargins(16, 12, 16, 12)
        fc.setSpacing(6)

        passes_info = ", ".join(self.shaders.keys()) if self.shaders else "(aucun shader chargé)"
        fc.addWidget(QLabel(f"Passes GLSL : <b>{passes_info}</b>"))

        self._chk_include_sources = QCheckBox("Inclure les sources GLSL (dossier shaders/)")
        self._chk_include_sources.setChecked(True)
        fc.addWidget(self._chk_include_sources)

        lbl_note = QLabel(
            "ℹ Le player HTML5/WebGL2 est auto-contenu (zéro dépendance CDN).\n"
            "  Hébergez simplement le dossier sur n'importe quel serveur statique\n"
            "  (GitHub Pages, Netlify, itch.io, votre propre hébergement…).")
        lbl_note.setWordWrap(True)
        lbl_note.setStyleSheet("color: palette(mid); font-size: 11px;")
        fc.addWidget(lbl_note)

        vl.addWidget(grp_content)

        # ── Section Dossier de sortie ────────────────────────────────────────
        grp_out = QGroupBox("Dossier de sortie")
        hl_out = QHBoxLayout(grp_out)
        hl_out.setContentsMargins(16, 12, 16, 12)
        hl_out.setSpacing(8)

        self._edit_output = QLineEdit()
        self._edit_output.setPlaceholderText("Choisir un dossier…")
        self._edit_output.setReadOnly(True)
        hl_out.addWidget(self._edit_output, 1)

        btn_browse = QPushButton("Parcourir…")
        btn_browse.clicked.connect(self._browse_output)
        hl_out.addWidget(btn_browse)

        vl.addWidget(grp_out)
        vl.addStretch()

        # ── Boutons ──────────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(sep)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(24, 12, 24, 16)
        btn_row.setSpacing(8)

        self._lbl_status = QLabel("")
        self._lbl_status.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        btn_row.addWidget(self._lbl_status, 1)

        btn_cancel = QPushButton("Annuler")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        self._btn_export = QPushButton("🌐 Générer la galerie")
        self._btn_export.setDefault(True)
        self._btn_export.clicked.connect(self._on_export)
        btn_row.addWidget(self._btn_export)

        root.addLayout(btn_row)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(
            self, "Choisir le dossier de sortie",
            os.path.expanduser("~"))
        if d:
            # Propose un sous-dossier nommé d'après le titre
            title_slug = re.sub(r'[^\w\-]', '_',
                                self._edit_title.text().strip() or 'gallery').lower()
            proposed = os.path.join(d, f'gallery_{title_slug}')
            self._edit_output.setText(proposed)
            self._output_dir = proposed

    def _on_export(self):
        # Validation
        title = self._edit_title.text().strip()
        if not title:
            self._set_status("⚠ Le titre est obligatoire.", error=True)
            self._edit_title.setFocus()
            return
        author = self._edit_author.text().strip()
        if not author:
            self._set_status("⚠ L'auteur est obligatoire.", error=True)
            self._edit_author.setFocus()
            return
        out_dir = self._edit_output.text().strip()
        if not out_dir:
            self._set_status("⚠ Choisissez un dossier de sortie.", error=True)
            return
        if not self.shaders:
            self._set_status("⚠ Aucun shader chargé.", error=True)
            return

        # Confirmation si le dossier existe déjà
        if os.path.exists(out_dir):
            reply = QMessageBox.question(
                self, "Dossier existant",
                f"Le dossier existe déjà :\n{out_dir}\n\nÉcraser son contenu ?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Métadonnées
        tags = [t.strip() for t in self._edit_tags.text().split(',') if t.strip()]
        meta = {
            'title':       title,
            'author':      author,
            'description': self._edit_desc.toPlainText().strip(),
            'tags':        tags,
            'licence':     self._cmb_licence.currentText(),
            'resolution':  self.resolution,
        }

        # Shaders à inclure
        shaders = self.shaders
        if not self._chk_include_sources.isChecked():
            # On exporte quand même le shader Image pour le player
            img_src = (shaders.get('Image') or shaders.get('image')
                       or next(iter(shaders.values()), ''))
            shaders = {'Image': img_src}

        preview = (self.preview_pixmap
                   if self._chk_include_preview.isChecked() else None)

        # Export
        self._btn_export.setEnabled(False)
        self._set_status("Export en cours…")
        try:
            exporter = GalleryExporter(meta, shaders, preview)
            files = exporter.export(out_dir)
            self._set_status(f"✓ {len(files)} fichiers générés.")
            self._save_settings()

            # Ouvrir le dossier + proposer d'ouvrir index.html
            import sys as _sys, subprocess as _sub
            index_path = os.path.join(out_dir, 'index.html')
            reply = QMessageBox.information(
                self, "Galerie générée",
                f"✓ {len(files)} fichiers créés dans :\n{out_dir}\n\n"
                "Ouvrir index.html dans le navigateur ?",
                QMessageBox.StandardButton.Open | QMessageBox.StandardButton.Close,
                QMessageBox.StandardButton.Open)
            if reply == QMessageBox.StandardButton.Open:
                import webbrowser
                webbrowser.open(Path(index_path).as_uri())

            self.accept()
        except OSError as e:
            self._set_status(f"❌ Erreur : {e}", error=True)
            log.exception("Erreur export galerie")
        finally:
            self._btn_export.setEnabled(True)

    def _set_status(self, msg: str, error: bool = False):
        self._lbl_status.setText(msg)
        color = "palette(highlighted-text)" if error else "palette(mid)"
        self._lbl_status.setStyleSheet(
            f"color: {'#e05050' if error else '#70a070'}; font-size: 11px;")

    # ── Persistance des champs ────────────────────────────────────────────────

    def _save_settings(self):
        s = QSettings("OpenShader", "OpenShader")
        s.setValue("gallery/author",  self._edit_author.text().strip())
        s.setValue("gallery/licence", self._cmb_licence.currentText())
        s.setValue("gallery/last_output_dir",
                   os.path.dirname(self._edit_output.text().strip()))

    def _load_settings(self):
        s = QSettings("OpenShader", "OpenShader")
        author = s.value("gallery/author", "")
        if author:
            self._edit_author.setText(author)
        licence = s.value("gallery/licence", "")
        if licence:
            idx = self._cmb_licence.findText(licence)
            if idx >= 0:
                self._cmb_licence.setCurrentIndex(idx)
        last_dir = s.value("gallery/last_output_dir", os.path.expanduser("~"))
        if last_dir and os.path.isdir(last_dir):
            # Pré-remplir avec un chemin basé sur le dernier dossier utilisé
            self._edit_output.setText(os.path.join(last_dir, 'gallery_untitled'))
            self._output_dir = self._edit_output.text()
