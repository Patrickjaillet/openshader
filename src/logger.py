"""
logger.py
---------
Logger centralisé pour OpenShader / DemoMaker.
Utilise le module stdlib `logging` avec un format coloré en console.

Usage :
    from .logger import get_logger
    log = get_logger(__name__)
    log.info("message")
    log.warning("attention")
    log.error("erreur")

Niveau par défaut : INFO.
Activer le mode debug : passer --debug en argument CLI ou appeler set_debug(True).
"""

import logging
import sys

_FMT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
_DATE = "%H:%M:%S"

# Codes ANSI pour la console
_COLORS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # vert
    "WARNING":  "\033[33m",   # jaune
    "ERROR":    "\033[31m",   # rouge
    "CRITICAL": "\033[35m",   # magenta
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{_RESET}"
        return super().format(record)


def _setup_root_logger():
    root = logging.getLogger("openshader")
    if root.handlers:
        return  # déjà configuré
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ColorFormatter(fmt=_FMT, datefmt=_DATE))
    root.addHandler(handler)
    root.propagate = False


_setup_root_logger()


def get_logger(name: str) -> logging.Logger:
    """Retourne un logger enfant de 'openshader'."""
    if not name.startswith("openshader"):
        name = f"openshader.{name.rsplit('.', 1)[-1]}"
    return logging.getLogger(name)


def set_debug(enabled: bool = True):
    """Active ou désactive le niveau DEBUG pour tous les loggers openshader."""
    level = logging.DEBUG if enabled else logging.INFO
    logging.getLogger("openshader").setLevel(level)
