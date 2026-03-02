#!/usr/bin/env python3
"""
main.py
-------
Point d'entrée du OpenShader.
Lance l'application PyQt6 ou effectue un rendu headless selon les arguments.

Dépendances requises :
    pip install PyQt6 moderngl pygame numpy

Usage :
    python main.py                              # mode GUI
    python main.py --debug                      # mode GUI avec logs détaillés
    python main.py --render --shader foo.st \\
                   --output video.mp4           # rendu headless (sans Qt)

Pour l'aide complète du mode rendu :
    python main.py --render --help
"""

import sys
import os

# Ajoute le dossier src au path Python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.logger import get_logger, set_debug

log = get_logger(__name__)


def _is_render_mode() -> bool:
    """Détecte si --render est dans les arguments AVANT de créer Qt."""
    return '--render' in sys.argv


def run_headless() -> int:
    """Mode rendu headless — pas de Qt, pas de fenêtre."""
    # Retire --render de argv pour que argparse headless ne le voit pas
    argv = [a for a in sys.argv[1:] if a != '--render']

    # Gestion --debug
    if '--debug' in argv:
        set_debug(True)
        argv = [a for a in argv if a != '--debug']
        log.debug("Mode debug activé")

    from src.headless import main_headless
    return main_headless(argv)


def run_gui() -> int:
    """Mode GUI standard — lance la fenêtre PyQt6."""
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore    import Qt, QCoreApplication
    from PyQt6.QtGui     import QSurfaceFormat
    from src.main_window import MainWindow

    if '--debug' in sys.argv:
        set_debug(True)
        sys.argv.remove('--debug')
        log.debug("Mode debug activé")

    QCoreApplication.setApplicationName("OpenShader")
    QCoreApplication.setApplicationVersion("1.0.0")
    QCoreApplication.setOrganizationName("Demoscene Tools")

    # ── HiDPI / Retina / 4K : doit être positionné AVANT QApplication ────────
    # PyQt6 active le HiDPI par défaut — on force seulement le rounding policy
    # pour les fractional scale factors (1.25×, 1.5×…) sur Windows/Linux.
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("QT_SCALE_FACTOR_ROUNDING_POLICY", "PassThrough")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSwapInterval(1)
    QSurfaceFormat.setDefaultFormat(fmt)

    window = MainWindow()
    window.show()
    return app.exec()

_CLI_COMMANDS = {"render", "export", "serve", "info"}


def _is_cli_mode() -> bool:
    """Détecte si la première commande est une commande CLI."""
    return len(sys.argv) > 1 and sys.argv[1] in _CLI_COMMANDS


def run_cli() -> int:
    """Mode CLI enrichi — délègue à src.cli."""
    from src.cli import main as cli_main
    return cli_main(sys.argv[1:])


def main():
    if _is_cli_mode():
        sys.exit(run_cli())
    elif _is_render_mode():
        sys.exit(run_headless())
    else:
        sys.exit(run_gui())


if __name__ == "__main__":
    main()
