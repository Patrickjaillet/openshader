"""
headless.py
-----------
Rendu headless (sans Qt, sans fenêtre) d'un shader vers un fichier vidéo ou PNG.

Usage CLI via main.py :
    python main.py --render --shader path/to/foo.st \\
                   --output video.mp4               \\
                   [--width 1280] [--height 720]     \\
                   [--fps 60] [--duration 10]        \\
                   [--pass Image]                    \\
                   [--define KEY=VALUE ...]          \\
                   [--lib-dir path/to/lib]           \\
                   [--loglevel debug|info|warning]

Formats de sortie supportés (déduits de l'extension) :
    .mp4  .mkv  .mov  .webm  → vidéo via FFmpeg
    .gif                     → GIF animé via FFmpeg
    .png                     → séquence PNG  (frame_%05d.png)
    (un seul .png sans %0Nd) → snapshot d'un seul frame à t=0
"""

from __future__ import annotations

import os
import sys
import subprocess
import time
import argparse
import logging
import struct
from pathlib import Path

log = logging.getLogger(__name__)


# ── Backend OpenGL offscreen ───────────────────────────────────────────────────

def _create_offscreen_context(width: int, height: int):
    """Crée un contexte ModernGL standalone (EGL > OSMesa > fallback Qt hidden)."""
    import moderngl

    # 1. EGL (Linux/Mesa, headless serveurs)
    try:
        ctx = moderngl.create_standalone_context(backend='egl')
        log.info("Contexte EGL créé")
        return ctx
    except (moderngl.Error, RuntimeError) as e:
        log.debug("EGL indisponible : %s", e)

    # 2. OSMesa (software rasterizer — toujours disponible si libOSMesa installé)
    try:
        ctx = moderngl.create_standalone_context(backend='osmesa')
        log.info("Contexte OSMesa créé")
        return ctx
    except (moderngl.Error, RuntimeError) as e:
        log.debug("OSMesa indisponible : %s", e)

    # 3. Fallback : contexte natif (nécessite un display, p.ex. Xvfb)
    try:
        ctx = moderngl.create_standalone_context()
        log.info("Contexte OpenGL natif créé")
        return ctx
    except (moderngl.Error, RuntimeError) as e:
        log.error("Impossible de créer un contexte OpenGL : %s", e)
        raise RuntimeError(
            "Aucun backend OpenGL offscreen disponible.\n"
            "Installer libEGL (Mesa) ou libOSMesa, ou utiliser Xvfb."
        ) from e


# ── FFmpeg pipe ───────────────────────────────────────────────────────────────

def _ffmpeg_available() -> bool:
    try:
        subprocess.run(['ffmpeg', '-version'], timeout=10,
                       capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _open_ffmpeg_pipe(output: str, width: int, height: int, fps: float) -> subprocess.Popen:
    ext = Path(output).suffix.lower()

    if ext == '.gif':
        # Deux passes (palette + encode) — on va tout faire en une seule passe
        # en utilisant palettegen+paletteuse via lavfi split
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'rgba',
            '-r', str(fps), '-i', 'pipe:0',
            '-vf', (f'vflip,fps={fps},'
                    f'split[s0][s1];[s0]palettegen=max_colors=256[p];[s1][p]paletteuse=dither=bayer'),
            output,
        ]
    elif ext == '.webm':
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'rgba',
            '-r', str(fps), '-i', 'pipe:0',
            '-vf', 'vflip',
            '-c:v', 'libvpx-vp9', '-pix_fmt', 'yuva420p',
            '-b:v', '0', '-crf', '30',
            output,
        ]
    else:  # mp4 / mkv / mov
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'rgba',
            '-r', str(fps), '-i', 'pipe:0',
            '-vf', 'vflip',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-preset', 'fast', '-crf', '18',
            output,
        ]
    log.debug("FFmpeg cmd : %s", ' '.join(cmd))
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)


# ── PNG writer (no Pillow needed) ─────────────────────────────────────────────

def _write_png(path: str, rgba: bytes, width: int, height: int):
    """Écrit un PNG RGBA sans dépendances externes (pur stdlib zlib)."""
    import zlib

    def _chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack('>I', len(data)) + tag + data
        return c + struct.pack('>I', zlib.crc32(tag + data) & 0xFFFFFFFF)

    # Flip vertical (OpenGL origin bas-gauche → PNG origine haut-gauche)
    row_size = width * 4
    rows = [rgba[i * row_size:(i + 1) * row_size]
            for i in range(height - 1, -1, -1)]

    # Filter type 0 (none) pour chaque ligne
    raw = b''.join(b'\x00' + row for row in rows)
    compressed = zlib.compress(raw, 6)

    png  = b'\x89PNG\r\n\x1a\n'
    png += _chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0))
    png += _chunk(b'IDAT', compressed)
    png += _chunk(b'IEND', b'')

    with open(path, 'wb') as f:
        f.write(png)


# ── Main render loop ──────────────────────────────────────────────────────────

def render_headless(
    shader_path: str,
    output: str,
    width: int = 800,
    height: int = 450,
    fps: float = 60.0,
    duration: float = 10.0,
    pass_name: str = 'Image',
    defines: dict | None = None,
    lib_dir: str | None = None,
) -> int:
    """
    Effectue le rendu headless et retourne le code de sortie (0 = succès).
    """
    t0 = time.perf_counter()

    # ── Valider les entrées ────────────────────────────────────────────────
    shader_path = os.path.abspath(shader_path)
    if not os.path.isfile(shader_path):
        log.error("Shader introuvable : %s", shader_path)
        return 1

    output = os.path.abspath(output)
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)

    ext = Path(output).suffix.lower()
    video_exts = {'.mp4', '.mkv', '.mov', '.webm', '.gif'}
    png_seq    = '%' in output and ext == '.png'
    single_png = not png_seq and ext == '.png'
    is_video   = ext in video_exts

    if is_video and not _ffmpeg_available():
        log.error("FFmpeg est requis pour l'export vidéo (introuvable dans le PATH).")
        return 1

    total_frames = max(1, int(duration * fps)) if not single_png else 1
    log.info("Rendu headless : %s → %s  [%dx%d @ %g fps, %.2fs, %d frames]",
             os.path.basename(shader_path), os.path.basename(output),
             width, height, fps, duration, total_frames)

    # ── Lire le shader ────────────────────────────────────────────────────
    with open(shader_path, encoding='utf-8') as f:
        source = f.read()

    # ── Créer le moteur ───────────────────────────────────────────────────
    # Import local pour ne pas dépendre de Qt
    _src_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_src_dir, '..'))
    from src.shader_engine import ShaderEngine

    ctx = _create_offscreen_context(width, height)

    engine = ShaderEngine(width=width, height=height, lib_dir=lib_dir)
    engine.initialize(ctx)

    ok, err = engine.load_shader_source(source, pass_name, source_path=shader_path)
    if not ok:
        log.error("Erreur de compilation GLSL :\n%s", err)
        engine.cleanup()
        ctx.release()
        return 1

    log.info("Shader compilé OK (pass '%s')", pass_name)

    # ── Ouvrir la sortie ──────────────────────────────────────────────────
    ffmpeg_proc = None
    if is_video:
        ffmpeg_proc = _open_ffmpeg_pipe(output, width, height, fps)

    # ── Boucle de rendu ───────────────────────────────────────────────────
    try:
        for frame_idx in range(total_frames):
            current_time = frame_idx / fps

            # Progress bar simple
            if total_frames > 1:
                pct = (frame_idx + 1) / total_frames
                bar_w = 40
                filled = int(bar_w * pct)
                bar = '█' * filled + '░' * (bar_w - filled)
                eta = (time.perf_counter() - t0) / max(pct, 1e-6) * (1.0 - pct)
                print(f'\r  [{bar}] {pct*100:5.1f}%  frame {frame_idx+1}/{total_frames}'
                      f'  ETA {eta:.0f}s   ', end='', flush=True)

            rgba = engine.render_frame(current_time)

            if single_png:
                _write_png(output, rgba, width, height)
            elif png_seq:
                frame_path = output % frame_idx
                _write_png(frame_path, rgba, width, height)
            elif is_video and ffmpeg_proc:
                ffmpeg_proc.stdin.write(rgba)

    except KeyboardInterrupt:
        print('\n  Annulé.')
        if ffmpeg_proc:
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()
        engine.cleanup()
        ctx.release()
        return 130

    finally:
        if total_frames > 1:
            print()  # newline après la barre de progression

    # ── Fermer FFmpeg ────────────────────────────────────────────────────
    if ffmpeg_proc:
        ffmpeg_proc.stdin.close()
        _, stderr = ffmpeg_proc.communicate()
        rc = ffmpeg_proc.returncode
        if rc != 0:
            log.error("FFmpeg a échoué (code %d) :\n%s",
                      rc, stderr.decode(errors='replace'))
            engine.cleanup()
            ctx.release()
            return 1

    engine.cleanup()
    ctx.release()

    elapsed = time.perf_counter() - t0
    log.info("Rendu terminé en %.2fs → %s", elapsed, output)
    print(f"  ✓  Rendu terminé en {elapsed:.2f}s  →  {output}")
    return 0


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python main.py --render',
        description='OpenShader — Rendu headless (sans interface graphique)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python main.py --render --shader shaders/glsl/aurora.glsl --output aurora.mp4
  python main.py --render --shader foo.st --output snapshot.png --duration 0
  python main.py --render --shader foo.st --output frames/frame_%05d.png \\
                 --width 1920 --height 1080 --fps 30 --duration 5
  python main.py --render --shader bar.st --output out.mp4 \\
                 --define HW_PERFORMANCE=1 --define MY_COLOR=vec3(1,0,0)
        """
    )
    p.add_argument('--shader',   required=True,  help='Chemin vers le fichier .st / .glsl / .trans')
    p.add_argument('--output',   required=True,  help='Fichier de sortie (.mp4 .mkv .mov .webm .gif .png)')
    p.add_argument('--width',    type=int,   default=800,  help='Largeur en pixels (défaut: 800)')
    p.add_argument('--height',   type=int,   default=450,  help='Hauteur en pixels (défaut: 450)')
    p.add_argument('--fps',      type=float, default=60.0, help='Images par seconde (défaut: 60)')
    p.add_argument('--duration', type=float, default=10.0, help='Durée en secondes (défaut: 10)')
    p.add_argument('--pass',     dest='pass_name', default='Image',
                   choices=['Image', 'Buffer A', 'Buffer B', 'Buffer C', 'Buffer D', 'Post'],
                   help="Passe à rendre (défaut: Image)")
    p.add_argument('--define',   action='append', default=[], metavar='KEY[=VALUE]',
                   help='Préprocesseur : #define à injecter (peut être répété)')
    p.add_argument('--lib-dir',  default=None,
                   help='Dossier des bibliothèques GLSL #include (défaut: shaders/lib/)')
    p.add_argument('--loglevel', default='info',
                   choices=['debug', 'info', 'warning', 'error'],
                   help='Niveau de log (défaut: info)')
    return p


def parse_defines(raw: list[str]) -> dict:
    """Convertit ['KEY=val', 'FLAG'] en {'KEY': 'val', 'FLAG': '1'}."""
    out = {}
    for item in raw:
        if '=' in item:
            k, v = item.split('=', 1)
            out[k.strip()] = v.strip()
        else:
            out[item.strip()] = '1'
    return out


def main_headless(argv: list[str] | None = None) -> int:
    """Point d'entrée appelé par main.py quand --render est présent."""
    parser = build_parser()
    args   = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper()),
        format='%(levelname)-8s %(name)s: %(message)s',
    )

    defines = parse_defines(args.define)

    return render_headless(
        shader_path = args.shader,
        output      = args.output,
        width       = args.width,
        height      = args.height,
        fps         = args.fps,
        duration    = args.duration,
        pass_name   = args.pass_name,
        defines     = defines,
        lib_dir     = args.lib_dir,
    )
