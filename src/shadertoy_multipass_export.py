"""
shadertoy_multipass_export.py
-----------------------------
v2.3 — Export Shadertoy multipass.

Génère le JSON Shadertoy complet depuis le Node Graph OpenShader :
  - Passe Common (uniforms partagés)
  - Passes Buffer A/B/C/D
  - Passe Image
  - Gestion des inputs iChannel0–3 (buffers ou textures)

Format de sortie : JSON Shadertoy v3 (compatible import manuel
et API Shadertoy : https://www.shadertoy.com/howto#q2)

Usage :
    from .shadertoy_multipass_export import ShadertoyExporter, show_multipass_export_dialog

    exporter = ShadertoyExporter(editors_dict, node_graph_dag)
    json_str = exporter.build_json()
    show_multipass_export_dialog(parent, json_str)
"""

from __future__ import annotations

import json
import re

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QCheckBox, QMessageBox, QApplication, QFileDialog,
)
from PyQt6.QtGui import QFont

from .logger import get_logger

log = get_logger(__name__)


# ── Shadertoy JSON schema helpers ─────────────────────────────────────────────

_ST_SAMPLER_DEFAULT = {
    "filter": "mipmap",
    "wrap":   "repeat",
    "vflip":  "true",
    "srgb":   "false",
    "internal": "byte",
}

_PASS_IDS = {
    "Image":    "4d",   # Image pass ID in Shadertoy JSON
    "Buffer A": "4e",
    "Buffer B": "4f",
    "Buffer C": "4g",
    "Buffer D": "4h",
    "Common":   "4c",
}

_BUFFER_CHANNEL_IDS = {
    "Buffer A": "4e",
    "Buffer B": "4f",
    "Buffer C": "4g",
    "Buffer D": "4h",
}


def _make_input(channel: int, src_pass: str | None, texture_url: str | None = None) -> dict:
    """Construit un objet input iChannel<channel>."""
    if src_pass and src_pass in _BUFFER_CHANNEL_IDS:
        return {
            "id":      channel,
            "src":     f"/media/previz/{_BUFFER_CHANNEL_IDS[src_pass]}.png",
            "ctype":   "buffer",
            "channel": channel,
            "sampler": _ST_SAMPLER_DEFAULT,
            "published": 1,
        }
    if texture_url:
        return {
            "id":      channel,
            "src":     texture_url,
            "ctype":   "texture",
            "channel": channel,
            "sampler": _ST_SAMPLER_DEFAULT,
            "published": 1,
        }
    return {}


def _make_output(pass_name: str) -> list[dict]:
    pid = _PASS_IDS.get(pass_name, "4d")
    return [{"id": 0, "channel": 0}]


def _shadertoy_source(source: str, pass_name: str, add_common_include: bool) -> str:
    """
    Nettoie et adapte le source GLSL pour Shadertoy :
      - Supprime le header OpenShader (#version, uniform iResolution…)
      - S'assure que mainImage est présente
      - Enveloppe un void main() si nécessaire
    """
    # Supprimer les lignes de header générées par le moteur
    lines = source.splitlines()
    clean = []
    _skip_uniforms = {
        "iResolution", "iTime", "iTimeDelta", "iFrame",
        "iMouse", "iChannel0", "iChannel1", "iChannel2", "iChannel3",
        "iChannelResolution", "uResolution", "uTime", "uTimeDelta", "uFrame",
        "_fragColor", "fragColor",
    }
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#version"):
            continue
        if stripped.startswith("out vec4"):
            continue
        if stripped.startswith("uniform") and any(u in stripped for u in _skip_uniforms):
            continue
        clean.append(line)

    result = "\n".join(clean).strip()

    # Si le shader utilise void main() → wrapper Shadertoy
    if "void main()" in result and "void mainImage" not in result:
        result = result.replace("out vec4 fragColor", "vec4 fragColor")
        result = result.replace("void main()", "_dm_main_impl()")
        result += (
            "\n\nvoid mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
            "    _dm_main_impl();\n"
            "}\n"
        )

    if add_common_include and pass_name != "Common":
        result = "// === Converti depuis OpenShader v2.3 ===\n\n" + result

    return result


# ── Exporter ──────────────────────────────────────────────────────────────────

class ShadertoyExporter:
    """
    Construit le JSON Shadertoy multipass depuis le projet OpenShader.

    Parameters
    ----------
    sources : dict[str, str]
        Dictionnaire pass_name → source GLSL brut (sans header).
        Passes reconnues : 'Image', 'Buffer A', 'Buffer B', 'Buffer C',
        'Buffer D', 'Common'.
    dag : dict[str, list[str]] | None
        Graphe DAG du Node Graph (source → liste de destinations).
        Utilisé pour résoudre les iChannel0–3.
    """

    def __init__(self, sources: dict[str, str], dag: dict | None = None):
        self.sources = {k: v for k, v in sources.items() if v and v.strip()}
        self.dag     = dag or {}

    def _resolve_inputs(self, pass_name: str) -> list[dict]:
        """
        Résout les inputs (iChannel0–3) d'une passe depuis le DAG.
        Le DAG est { src_node: [dst_node, …] }.
        Pour trouver les inputs d'une passe, on cherche qui pointe vers elle.
        """
        inputs = []
        channel = 0
        # Cherche les arêtes qui arrivent sur pass_name
        for src, dsts in self.dag.items():
            if pass_name in dsts and src in _BUFFER_CHANNEL_IDS:
                inp = _make_input(channel, src)
                if inp:
                    inputs.append(inp)
                    channel += 1
                if channel >= 4:
                    break
        return inputs

    def build_json(self, title: str = "OpenShader Export",
                   description: str = "Exporté depuis OpenShader v2.3") -> str:
        """Retourne le JSON Shadertoy complet (format v3)."""

        has_common = bool(self.sources.get("Common", "").strip())
        passes = []

        # Passe Common (si présente)
        if has_common:
            passes.append({
                "name":    "Common",
                "type":    "common",
                "code":    _shadertoy_source(self.sources["Common"], "Common", False),
                "description": "",
                "inputs":  [],
                "outputs": [],
            })

        # Passes Buffer A–D
        for buf in ("Buffer A", "Buffer B", "Buffer C", "Buffer D"):
            if buf not in self.sources:
                continue
            passes.append({
                "name":    buf,
                "type":    "buffer",
                "code":    _shadertoy_source(self.sources[buf], buf, has_common),
                "description": "",
                "inputs":  self._resolve_inputs(buf),
                "outputs": _make_output(buf),
            })

        # Passe Image (obligatoire)
        image_src = self.sources.get("Image", "void mainImage(out vec4 f, in vec2 c){ f=vec4(0); }")
        passes.append({
            "name":    "Image",
            "type":    "image",
            "code":    _shadertoy_source(image_src, "Image", has_common),
            "description": "",
            "inputs":  self._resolve_inputs("Image"),
            "outputs": _make_output("Image"),
        })

        shader = {
            "ver": "0.1",
            "info": {
                "id":          "",
                "date":        "0",
                "viewed":      0,
                "name":        title,
                "username":    "",
                "description": description,
                "likes":       0,
                "published":   0,
                "flags":       0,
                "tags":        ["demomaker"],
                "hasliked":    0,
            },
            "renderpass": passes,
        }
        return json.dumps({"Shader": shader}, indent=2, ensure_ascii=False)

    def build_clipboard_json(self) -> str:
        """Retourne uniquement la liste renderpass (pour paste dans Shadertoy)."""
        full = json.loads(self.build_json())
        return json.dumps(full["Shader"]["renderpass"], indent=2, ensure_ascii=False)


# ── Dialog ────────────────────────────────────────────────────────────────────

def show_multipass_export_dialog(parent, sources: dict[str, str],
                                  dag: dict | None = None):
    """
    Affiche le dialog d'export Shadertoy multipass.
    Permet copier dans le clipboard ou sauvegarder en fichier .json.
    """
    exporter = ShadertoyExporter(sources, dag)

    dlg = QDialog(parent)
    dlg.setWindowTitle("Export Shadertoy Multipass — v2.3")
    dlg.resize(700, 580)

    lay = QVBoxLayout(dlg)

    # En-tête
    info = QLabel(
        "Le JSON ci-dessous est au format Shadertoy v3.\n"
        "Collez-le dans l'API Shadertoy ou importez-le manuellement pass par pass."
    )
    info.setStyleSheet("color: #aaa; font-size: 11px;")
    lay.addWidget(info)

    # Passes détectées
    detected = [p for p in ("Common", "Buffer A", "Buffer B", "Buffer C", "Buffer D", "Image")
                if p in exporter.sources]
    passes_label = QLabel("Passes détectées : " + " · ".join(detected))
    passes_label.setStyleSheet("color: #7cf; font-weight: bold;")
    lay.addWidget(passes_label)

    # Mode clipboard (renderpass seulement vs JSON complet)
    cb_mode = QCheckBox("Mode clipboard (renderpass uniquement, pour coller dans Shadertoy)")
    cb_mode.setChecked(True)
    lay.addWidget(cb_mode)

    # Éditeur JSON
    te = QTextEdit()
    te.setReadOnly(True)
    te.setFont(QFont("Monospace", 10))
    te.setPlainText(exporter.build_clipboard_json())
    lay.addWidget(te)

    def _refresh():
        if cb_mode.isChecked():
            te.setPlainText(exporter.build_clipboard_json())
        else:
            te.setPlainText(exporter.build_json())

    cb_mode.toggled.connect(_refresh)

    # Boutons
    btn_row = QHBoxLayout()

    btn_copy = QPushButton("📋 Copier dans le presse-papiers")
    def _copy():
        QApplication.clipboard().setText(te.toPlainText())
        btn_copy.setText("✓ Copié !")
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(2000, lambda: btn_copy.setText("📋 Copier dans le presse-papiers"))
    btn_copy.clicked.connect(_copy)

    btn_save = QPushButton("💾 Enregistrer en .json…")
    def _save():
        path, _ = QFileDialog.getSaveFileName(
            dlg, "Enregistrer JSON Shadertoy", "shadertoy_export.json",
            "JSON Shadertoy (*.json)")
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(te.toPlainText())
                QMessageBox.information(dlg, "Enregistré", f"JSON sauvegardé :\n{path}")
            except OSError as e:
                QMessageBox.critical(dlg, "Erreur", str(e))
    btn_save.clicked.connect(_save)

    btn_open = QPushButton("🌐 Ouvrir Shadertoy…")
    def _open_st():
        import webbrowser
        webbrowser.open("https://www.shadertoy.com/new")
    btn_open.clicked.connect(_open_st)

    btn_close = QPushButton("Fermer")
    btn_close.clicked.connect(dlg.accept)

    btn_row.addWidget(btn_copy)
    btn_row.addWidget(btn_save)
    btn_row.addWidget(btn_open)
    btn_row.addStretch()
    btn_row.addWidget(btn_close)
    lay.addLayout(btn_row)

    dlg.exec()
