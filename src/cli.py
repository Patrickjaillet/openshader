"""
cli.py
------
CLI enrichie pour OpenShader — v1.0

Commandes disponibles :
  openshader render   — rendu headless vers fichier vidéo/PNG
  openshader export   — export multiformat (standalone, WASM, shadertoy…)
  openshader serve    — démarre le serveur REST local (FastAPI)
  openshader info     — affiche les infos d'un projet .st

Point d'entrée pip :
  [project.scripts]
  openshader = "src.cli:main"

Usage direct :
  python -m src.cli render --shader foo.st --output out.mp4
  python -m src.cli serve --port 8765
  python -m src.cli export --shader foo.st --format shadertoy
"""

from __future__ import annotations

import argparse
import sys
import os
import json
import logging

# Ajoute le dossier parent au path si exécuté directement
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

log = logging.getLogger("openshader.cli")


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _setup_logging(level: str = "info"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)-8s %(name)s: %(message)s",
    )


def _print_banner():
    print("┌─────────────────────────────────────────────┐")
    print("│  OpenShader CLI  v1.0                      │")
    print("└─────────────────────────────────────────────┘")


def _parse_defines(raw: list[str]) -> dict:
    out = {}
    for item in raw:
        if "=" in item:
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()
        else:
            out[item.strip()] = "1"
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  openshader render
# ═════════════════════════════════════════════════════════════════════════════

def _cmd_render(args: argparse.Namespace) -> int:
    """
    Rendu headless d'un shader vers un fichier vidéo ou image.
    Délègue à src.headless.render_headless.
    """
    _setup_logging(args.loglevel)
    from src.headless import render_headless

    defines = _parse_defines(args.define or [])

    print(f"  Shader  : {args.shader}")
    print(f"  Output  : {args.output}")
    print(f"  Taille  : {args.width}×{args.height} @ {args.fps} fps")
    print(f"  Durée   : {args.duration}s")
    if defines:
        print(f"  Defines : {defines}")
    print()

    return render_headless(
        shader_path=args.shader,
        output=args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration=args.duration,
        pass_name=args.pass_name,
        defines=defines,
        lib_dir=args.lib_dir,
    )


def _add_render_parser(sub: argparse._SubParsersAction):
    p = sub.add_parser(
        "render",
        help="Rendu headless d'un shader (sans interface graphique)",
        description="Rend un shader GLSL vers un fichier vidéo ou image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  openshader render --shader aurora.glsl --output aurora.mp4
  openshader render --shader foo.st --output snap.png --duration 0
  openshader render --shader bar.st --output out.mp4 \\
                     --width 1920 --height 1080 --fps 30 --duration 5
  openshader render --shader foo.st --output video.mp4 \\
                     --define HW_PERFORMANCE=1
        """,
    )
    p.add_argument("--shader",   required=True,  help="Chemin .st / .glsl")
    p.add_argument("--output",   required=True,  help="Sortie .mp4 .mkv .webm .gif .png")
    p.add_argument("--width",    type=int,   default=800,  help="Largeur px (défaut: 800)")
    p.add_argument("--height",   type=int,   default=450,  help="Hauteur px (défaut: 450)")
    p.add_argument("--fps",      type=float, default=60.0, help="FPS (défaut: 60)")
    p.add_argument("--duration", type=float, default=10.0, help="Durée s (défaut: 10)")
    p.add_argument("--pass",     dest="pass_name", default="Image",
                   choices=["Image", "Buffer A", "Buffer B", "Buffer C", "Buffer D", "Post"],
                   help="Passe à rendre (défaut: Image)")
    p.add_argument("--define",   action="append", default=[], metavar="KEY[=VALUE]",
                   help="#define à injecter (répétable)")
    p.add_argument("--lib-dir",  default=None, help="Dossier GLSL #include")
    p.add_argument("--loglevel", default="info",
                   choices=["debug", "info", "warning", "error"])
    p.set_defaults(func=_cmd_render)


# ═════════════════════════════════════════════════════════════════════════════
#  openshader export
# ═════════════════════════════════════════════════════════════════════════════

_EXPORT_FORMATS = ["shadertoy", "standalone", "wasm", "video", "png-seq"]


def _cmd_export(args: argparse.Namespace) -> int:
    """Export multiformat d'un projet OpenShader."""
    _setup_logging(args.loglevel)

    fmt = args.format
    shader = args.shader
    output = args.output or os.path.splitext(shader)[0]

    print(f"  Format  : {fmt}")
    print(f"  Shader  : {shader}")
    print(f"  Output  : {output}")
    print()

    if fmt == "video":
        # Délègue au moteur headless
        from src.headless import render_headless
        out_file = output if output.endswith((".mp4", ".mkv", ".webm", ".gif")) \
                   else output + ".mp4"
        return render_headless(
            shader_path=shader,
            output=out_file,
            width=args.width,
            height=args.height,
            fps=args.fps,
            duration=args.duration,
        )

    if fmt == "png-seq":
        from src.headless import render_headless
        os.makedirs(output, exist_ok=True)
        return render_headless(
            shader_path=shader,
            output=os.path.join(output, "frame_%05d.png"),
            width=args.width,
            height=args.height,
            fps=args.fps,
            duration=args.duration,
        )

    if fmt == "shadertoy":
        try:
            from src.shadertoy_multipass_export import export_shadertoy_url
            url = export_shadertoy_url(shader)
            out_file = output + "_shadertoy.json" if not output.endswith(".json") \
                       else output
            with open(out_file, "w") as f:
                json.dump({"url": url}, f, indent=2)
            print(f"  ✓  Export Shadertoy → {out_file}")
            return 0
        except Exception as e:
            log.error("Export Shadertoy : %s", e)
            return 1

    if fmt == "wasm":
        try:
            from src.wasm_exporter import export_wasm
            export_wasm(shader, output)
            print(f"  ✓  Export WASM → {output}")
            return 0
        except Exception as e:
            log.error("Export WASM : %s", e)
            return 1

    if fmt == "standalone":
        print("  ⚠  Export standalone nécessite PyInstaller.")
        print("     pip install pyinstaller  puis  python build.py")
        return 1

    print(f"  ✗  Format inconnu : {fmt}")
    return 1


def _add_export_parser(sub: argparse._SubParsersAction):
    p = sub.add_parser(
        "export",
        help="Export multiformat (video, png-seq, shadertoy, wasm, standalone)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Formats disponibles :
  video       → fichier vidéo MP4 via FFmpeg
  png-seq     → séquence PNG dans un dossier
  shadertoy   → JSON compatible Shadertoy
  wasm        → bundle WebAssembly
  standalone  → exécutable autonome (PyInstaller)

Exemples :
  openshader export --shader aurora.st --format video --output aurora.mp4
  openshader export --shader foo.st    --format png-seq --output frames/
  openshader export --shader bar.glsl  --format shadertoy
        """,
    )
    p.add_argument("--shader",   required=True,  help="Chemin .st / .glsl")
    p.add_argument("--format",   required=True,  choices=_EXPORT_FORMATS,
                   help=f"Format : {', '.join(_EXPORT_FORMATS)}")
    p.add_argument("--output",   default=None,   help="Fichier/dossier de sortie")
    p.add_argument("--width",    type=int,   default=1920)
    p.add_argument("--height",   type=int,   default=1080)
    p.add_argument("--fps",      type=float, default=60.0)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--loglevel", default="info",
                   choices=["debug", "info", "warning", "error"])
    p.set_defaults(func=_cmd_export)


# ═════════════════════════════════════════════════════════════════════════════
#  openshader serve
# ═════════════════════════════════════════════════════════════════════════════

def _cmd_serve(args: argparse.Namespace) -> int:
    """
    Démarre le serveur REST OpenShader en mode standalone.
    Nécessite fastapi + uvicorn. Peut se connecter à une instance en cours
    via un socket IPC, ou démarrer un stub si l'app n'est pas ouverte.
    """
    _setup_logging(args.loglevel)

    try:
        import fastapi
        import uvicorn
    except ImportError:
        print("  ✗  fastapi et uvicorn requis :")
        print("     pip install fastapi uvicorn")
        return 1

    host = args.host
    port = args.port

    print(f"  Démarrage du serveur REST OpenShader…")
    print(f"  URL  : http://{host}:{port}")
    print(f"  Docs : http://{host}:{port}/docs")
    print()
    print("  Ctrl+C pour arrêter.")
    print()

    # ── Stub app si OpenShader n'est pas ouvert ──────────────────────────────
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    stub_app = FastAPI(
        title="OpenShader REST API (stub)",
        description="OpenShader n'est pas ouvert — endpoints en lecture seule.",
        version="1.0.0",
    )

    @stub_app.get("/health")
    def health():
        return {"ok": True, "mode": "stub", "timestamp": __import__("time").time()}

    @stub_app.get("/status")
    def status():
        return {"error": "OpenShader n'est pas ouvert en mode GUI"}

    @stub_app.get("/docs")
    def docs_redirect():
        return JSONResponse({"endpoints": [
            "/health", "/status", "/play", "/pause", "/stop",
            "/seek", "/load", "/render",
            "/uniforms", "/timeline",
            "/shader/{pass_name}",
        ]})

    uvicorn.run(stub_app, host=host, port=port, log_level="warning")
    return 0


def _add_serve_parser(sub: argparse._SubParsersAction):
    p = sub.add_parser(
        "serve",
        help="Démarre le serveur REST local (FastAPI)",
        description="Expose OpenShader via HTTP/JSON pour le contrôle externe.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  openshader serve
  openshader serve --port 9000
  openshader serve --host 0.0.0.0 --port 8765

Note : pour contrôler une instance GUI ouverte, démarrez le serveur
depuis le menu OpenShader → Vue → API REST (Ctrl+Shift+R).
        """,
    )
    p.add_argument("--host",     default="127.0.0.1", help="Adresse d'écoute (défaut: 127.0.0.1)")
    p.add_argument("--port",     type=int, default=8765, help="Port (défaut: 8765)")
    p.add_argument("--loglevel", default="info",
                   choices=["debug", "info", "warning", "error"])
    p.set_defaults(func=_cmd_serve)


# ═════════════════════════════════════════════════════════════════════════════
#  openshader info
# ═════════════════════════════════════════════════════════════════════════════

def _cmd_info(args: argparse.Namespace) -> int:
    """Affiche les métadonnées d'un projet .st / .glsl."""
    _setup_logging(args.loglevel)

    path = args.shader
    if not os.path.isfile(path):
        print(f"  ✗  Fichier introuvable : {path}")
        return 1

    ext = os.path.splitext(path)[1].lower()
    size = os.path.getsize(path)

    print(f"  Fichier  : {os.path.abspath(path)}")
    print(f"  Taille   : {size} octets")
    print(f"  Type     : {ext}")

    if ext == ".st":
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            passes = list(data.get("passes", {}).keys()) or \
                     list(data.get("shaders", {}).keys())
            print(f"  Passes   : {', '.join(passes) if passes else '(aucune)'}")
            meta = data.get("meta", {})
            if meta:
                for k, v in meta.items():
                    print(f"  {k:<10}: {v}")
        except json.JSONDecodeError:
            print("  (fichier .st non JSON — format binaire ou legacy)")
        except Exception as e:
            print(f"  ✗  Erreur lecture : {e}")

    elif ext in (".glsl", ".frag", ".vert"):
        with open(path, encoding="utf-8", errors="replace") as f:
            src = f.read()
        lines = src.count("\n") + 1
        uniforms = __import__("re").findall(r"uniform\s+\w+\s+(\w+)", src)
        functions = __import__("re").findall(r"\b\w+\s+(\w+)\s*\(", src)
        print(f"  Lignes   : {lines}")
        print(f"  Uniforms : {', '.join(uniforms) if uniforms else '(aucun)'}")

    return 0


def _add_info_parser(sub: argparse._SubParsersAction):
    p = sub.add_parser(
        "info",
        help="Affiche les infos d'un projet .st / .glsl",
    )
    p.add_argument("shader", help="Chemin du fichier .st / .glsl")
    p.add_argument("--loglevel", default="warning",
                   choices=["debug", "info", "warning", "error"])
    p.set_defaults(func=_cmd_info)


# ═════════════════════════════════════════════════════════════════════════════
#  Parseur principal
# ═════════════════════════════════════════════════════════════════════════════

def build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="openshader",
        description="OpenShader CLI — contrôle et rendu de shaders GLSL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commandes disponibles :
  render     Rendu headless (sans fenêtre)
  export     Export multiformat
  serve      Serveur REST local
  info       Infos d'un projet

  openshader <commande> --help   pour l'aide détaillée de chaque commande
        """,
    )
    p.add_argument("--version", action="version", version="OpenShader CLI 1.0")
    sub = p.add_subparsers(dest="command", metavar="<commande>")
    sub.required = True
    _add_render_parser(sub)
    _add_export_parser(sub)
    _add_serve_parser(sub)
    _add_info_parser(sub)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
