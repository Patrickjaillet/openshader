"""
standalone_player.py
--------------------
v2.3 — Lecteur démo standalone.

Génère un dossier (ou zip) autonome contenant :
  - Un script Python minimal (~200 lignes) sans dépendance Qt
  - Les shaders du projet (.st / .glsl)
  - Le fichier audio (si présent)
  - Le projet .demomaker
  - Un requirements_player.txt (moderngl + pygame uniquement)
  - Un README_PLAYER.md

Le lecteur standalone :
  - Charge et joue le projet en plein-écran via pygame + moderngl
  - Supporte la timeline (évaluation des uniforms)
  - Pas de GUI : Espace = pause, Échap = quitter
  - Cible : soumissions compo Demoscene

Usage :
    from .standalone_player import StandaloneExporter
    exp = StandaloneExporter(project_data, shaders_dir, audio_path)
    exp.export_to(output_dir)
"""

from __future__ import annotations

import os
import shutil
import zipfile
import json
from pathlib import Path

from .logger import get_logger

log = get_logger(__name__)


# ── Player source template ─────────────────────────────────────────────────────

_PLAYER_PY = '''\
#!/usr/bin/env python3
"""
OpenShader Standalone Player — v2.3
Généré automatiquement. Ne pas modifier manuellement.

Dépendances : pip install moderngl pygame numpy
Lancement   : python player.py [--fullscreen] [--width W] [--height H]
"""

import sys, os, json, time, argparse
import numpy as np

# ── Chargement du projet ──────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

def _load_project():
    for name in ("project.demomaker", "project.dmk", "project.json"):
        p = os.path.join(_HERE, name)
        if os.path.isfile(p):
            import zipfile as _z
            try:
                if name.endswith(".demomaker") and _z.is_zipfile(p):
                    with _z.ZipFile(p) as zf:
                        return json.loads(zf.read("project.json").decode())
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, KeyError, OSError, _z.BadZipFile) as e:
                import sys
                print(f"[standalone_player] Impossible de lire {name}: {e}", file=sys.stderr)
                return {}
    return {}

# ── Préprocesseur GLSL minimal ────────────────────────────────────────────────
import re as _re

def _preprocess(src):
    out = []
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("#version") or s.startswith("uniform") and any(
            u in s for u in ("iResolution","iTime","iTimeDelta","iFrame",
                             "iMouse","uResolution","uTime","_fragColor","fragColor")):
            out.append("")
        else:
            out.append(line)
    return "\\n".join(out)

VERTEX = """
#version 330 core
in vec2 in_position;
void main(){ gl_Position = vec4(in_position, 0.0, 1.0); }
"""

def _build_shader(source):
    is_st = bool(_re.search(r"void\\s+mainImage\\s*\\(", source))
    clean = _re.sub(r"#version\\s+\\d+\\s*(core|compatibility)?\\s*\\n", "", source)
    if is_st:
        header = """
#version 330 core
uniform vec3  iResolution;
uniform float iTime;
uniform float iTimeDelta;
uniform int   iFrame;
uniform vec4  iMouse;
out vec4 _fragColor;
"""
        footer = "\\nvoid main(){ mainImage(_fragColor, gl_FragCoord.xy); }\\n"
        return header + clean + footer
    else:
        header = """
#version 330 core
uniform vec2  uResolution;
uniform float uTime;
uniform float uTimeDelta;
uniform int   uFrame;
out vec4 fragColor;
"""
        if clean.strip().startswith("#version"):
            return clean
        return header + clean

# ── Timeline évaluation minimale ──────────────────────────────────────────────

def _lerp(a, b, t): return a + (b - a) * t
def _smoothstep(a, b, t):
    t = max(0.0, min(1.0, (t - a) / (b - a))) if b != a else 0.0
    return t * t * (3 - 2 * t)

def _eval_track(track, t):
    kfs = sorted(track.get("keyframes", []), key=lambda k: k["time"])
    if not kfs: return track.get("default_value", 0.0)
    if t <= kfs[0]["time"]: return kfs[0]["value"]
    if t >= kfs[-1]["time"]: return kfs[-1]["value"]
    for i in range(len(kfs) - 1):
        a, b = kfs[i], kfs[i+1]
        if a["time"] <= t <= b["time"]:
            u = (t - a["time"]) / max(b["time"] - a["time"], 1e-9)
            interp = a.get("interp", "linear")
            if interp == "step":   return a["value"]
            if interp == "smooth": u = u*u*(3-2*u)
            av, bv = a["value"], b["value"]
            if isinstance(av, list):
                return [_lerp(av[j], bv[j], u) for j in range(len(av))]
            return _lerp(av, bv, u)
    return kfs[-1]["value"]

def _evaluate_timeline(project, t):
    uniforms = {}
    for track in project.get("timeline", {}).get("tracks", []):
        name = track.get("uniform_name") or track.get("name")
        if name:
            uniforms[name] = _eval_track(track, t)
    return uniforms

# ── Rendu principal ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    project = _load_project()

    # Shader principal (Image pass)
    shader_path = None
    shader_source = ""
    for pass_data in project.get("passes", {}).values():
        if isinstance(pass_data, dict):
            sp = pass_data.get("source_path") or pass_data.get("file")
            if sp:
                sp = os.path.join(_HERE, os.path.basename(sp))
                if os.path.isfile(sp):
                    shader_path = sp
                    break
        elif isinstance(pass_data, str):
            # Inline source
            shader_source = pass_data
            break

    if shader_path and not shader_source:
        with open(shader_path, encoding="utf-8") as f:
            shader_source = f.read()

    if not shader_source:
        # Shader de fallback
        shader_source = """void mainImage(out vec4 f, in vec2 c){
    vec2 uv = c / iResolution.xy;
    f = vec4(uv, 0.5+0.5*sin(iTime), 1.0);
}"""

    # pygame + moderngl
    import pygame
    import moderngl

    pygame.init()
    W, H = args.width, args.height
    flags = pygame.OPENGL | pygame.DOUBLEBUF | (pygame.FULLSCREEN if args.fullscreen else 0)
    pygame.display.set_mode((W, H), flags, vsync=1)
    pygame.display.set_caption("OpenShader Player")
    ctx = moderngl.create_context()

    frag = _build_shader(shader_source)
    try:
        prog = ctx.program(vertex_shader=VERTEX, fragment_shader=frag)
    except moderngl.Error as e:
        print("Erreur compilation shader:", e)
        sys.exit(1)

    verts = np.array([-1,-1, 1,-1, -1,1, 1,-1, 1,1, -1,1], dtype="f4")
    vbo   = ctx.buffer(verts)
    vao   = ctx.vertex_array(prog, [(vbo, "2f", "in_position")])

    def _s(n, v):
        try:
            if n in prog: prog[n].value = v
        except Exception: pass

    # Audio
    duration = project.get("timeline", {}).get("duration", 30.0)
    audio_files = [f for f in os.listdir(_HERE)
                   if f.endswith((".wav", ".mp3", ".ogg"))]
    if audio_files:
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(os.path.join(_HERE, audio_files[0]))
        except Exception as ae:
            print("Audio non disponible:", ae)

    t_start  = time.perf_counter()
    paused   = False
    pause_t  = 0.0
    frame    = 0
    clock    = pygame.time.Clock()

    # Démarrage audio
    if audio_files:
        try: pygame.mixer.music.play()
        except Exception: pass

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit(0)
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit(0)
                if ev.key == pygame.K_SPACE:
                    paused = not paused
                    if paused:
                        pause_t = time.perf_counter()
                        try: pygame.mixer.music.pause()
                        except Exception: pass
                    else:
                        t_start += time.perf_counter() - pause_t
                        try: pygame.mixer.music.unpause()
                        except Exception: pass

        if paused:
            clock.tick(30)
            continue

        t = time.perf_counter() - t_start
        if t > duration:
            t_start = time.perf_counter()
            frame   = 0
            t       = 0.0
            if audio_files:
                try: pygame.mixer.music.rewind(); pygame.mixer.music.play()
                except Exception: pass

        # Uniforms timeline
        uniforms = _evaluate_timeline(project, t)
        for name, val in uniforms.items():
            _s(name, val)

        # Uniforms système
        _s("iResolution", (float(W), float(H), 1.0))
        _s("iTime",       float(t))
        _s("iFrame",      frame)
        mx, my = pygame.mouse.get_pos()
        mb = pygame.mouse.get_pressed()
        _s("iMouse", (float(mx), float(H - my), float(mx) if mb[0] else 0.0, float(H - my) if mb[0] else 0.0))
        _s("uResolution", (float(W), float(H)))
        _s("uTime",       float(t))
        _s("uFrame",      frame)

        ctx.clear(0, 0, 0, 1)
        vao.render()
        pygame.display.flip()
        clock.tick(60)
        frame += 1

if __name__ == "__main__":
    main()
'''

_REQUIREMENTS_PLAYER = """\
# OpenShader Standalone Player — dépendances minimales
moderngl>=5.8
pygame>=2.5
numpy>=1.24
"""

_README_PLAYER = """\
# OpenShader Standalone Player

Généré par OpenShader v2.3.

## Lancement

```bash
pip install -r requirements_player.txt
python player.py
```

## Options

```
python player.py --fullscreen        # Plein-écran
python player.py --width 1920 --height 1080
```

## Contrôles

| Touche  | Action |
|---------|--------|
| Espace  | Pause / Reprendre |
| Échap   | Quitter |

## Contenu

- `player.py`            — Lecteur autonome (~200 lignes)
- `project.demomaker`    — Projet complet
- `requirements_player.txt`
- Shaders `.st` / `.glsl`
- Audio `.wav` / `.mp3` / `.ogg` (si inclus)
"""


# ── Exporter ──────────────────────────────────────────────────────────────────

class StandaloneExporter:
    """
    Exporte le projet OpenShader en lecteur standalone autonome.

    Parameters
    ----------
    project_data : dict
        Données du projet (structure .demomaker JSON).
    project_path : str | None
        Chemin du fichier .demomaker source (pour copier les assets).
    audio_path : str | None
        Chemin du fichier audio actif.
    shaders : dict[str, str]
        Dictionnaire pass_name → source GLSL brut.
    """

    def __init__(self, project_data: dict, project_path: str | None = None,
                 audio_path: str | None = None, shaders: dict | None = None):
        self.project_data = project_data
        self.project_path = project_path
        self.audio_path   = audio_path
        self.shaders      = shaders or {}

    def export_to_dir(self, output_dir: str) -> list[str]:
        """
        Exporte vers output_dir. Retourne la liste des fichiers créés.
        """
        os.makedirs(output_dir, exist_ok=True)
        created = []

        # player.py
        player_path = os.path.join(output_dir, "player.py")
        with open(player_path, "w", encoding="utf-8") as f:
            f.write(_PLAYER_PY)
        created.append(player_path)

        # requirements_player.txt
        req_path = os.path.join(output_dir, "requirements_player.txt")
        with open(req_path, "w", encoding="utf-8") as f:
            f.write(_REQUIREMENTS_PLAYER)
        created.append(req_path)

        # README_PLAYER.md
        readme_path = os.path.join(output_dir, "README_PLAYER.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(_README_PLAYER)
        created.append(readme_path)

        # Copie du .demomaker
        if self.project_path and os.path.isfile(self.project_path):
            dst = os.path.join(output_dir, "project.demomaker")
            shutil.copy2(self.project_path, dst)
            created.append(dst)
        else:
            # Sérialise depuis les données en mémoire
            dst = os.path.join(output_dir, "project.json")
            with open(dst, "w", encoding="utf-8") as f:
                import json
                json.dump(self.project_data, f, indent=2, ensure_ascii=False)
            created.append(dst)

        # Shaders inline → fichiers
        for pass_name, source in self.shaders.items():
            if not source or not source.strip():
                continue
            safe_name = pass_name.lower().replace(" ", "_")
            ext       = ".st" if "mainImage" in source else ".glsl"
            shader_dst = os.path.join(output_dir, f"{safe_name}{ext}")
            with open(shader_dst, "w", encoding="utf-8") as f:
                f.write(source)
            created.append(shader_dst)

        # Audio
        if self.audio_path and os.path.isfile(self.audio_path):
            audio_dst = os.path.join(output_dir, os.path.basename(self.audio_path))
            shutil.copy2(self.audio_path, audio_dst)
            created.append(audio_dst)

        log.info("Standalone exporté dans %s (%d fichiers)", output_dir, len(created))
        return created

    def export_to_zip(self, zip_path: str) -> int:
        """
        Exporte dans un ZIP autonome. Retourne le nombre de fichiers.
        """
        import tempfile
        with tempfile.TemporaryDirectory(prefix="dm_standalone_") as tmp:
            files = self.export_to_dir(tmp)
            base  = Path(zip_path).stem
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
                for f in files:
                    zf.write(f, arcname=os.path.join(base, os.path.basename(f)))
            log.info("ZIP standalone : %s (%d fichiers)", zip_path, len(files))
            return len(files)
