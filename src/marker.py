"""
marker.py
---------
Modèle de données pour la piste de marqueurs.
Les marqueurs sont des repères nommés et colorés posés sur la timeline
(intro, drop, build, outro, etc.).

v1.2 — Nouveau module.
"""

from __future__ import annotations
import bisect
from dataclasses import dataclass, field


# ── Modèle ───────────────────────────────────────────────────────────────────

@dataclass
class Marker:
    """Un marqueur positionné sur la timeline."""
    time:  float
    label: str  = ""
    color: str  = "#F59E0B"   # amber par défaut


class MarkerTrack:
    """
    Piste spéciale de marqueurs, indépendante des pistes uniformes.
    Affichée entre la règle et les pistes normales.
    """

    def __init__(self):
        self.markers: list[Marker] = []

    # ── CRUD ────────────────────────────────────────────────────────────────

    def add(self, t: float, label: str = "", color: str = "#F59E0B") -> Marker:
        """Ajoute un marqueur (trié par temps). Remplace si ±0.01s."""
        for i, m in enumerate(self.markers):
            if abs(m.time - t) < 0.01:
                self.markers[i] = Marker(t, label, color)
                return self.markers[i]
        m   = Marker(t, label, color)
        idx = bisect.bisect_left([mk.time for mk in self.markers], t)
        self.markers.insert(idx, m)
        return m

    def remove(self, marker: Marker):
        """Supprime un marqueur (par identité)."""
        if marker in self.markers:
            self.markers.remove(marker)

    def nearest(self, t: float, tol: float = 0.2) -> Marker | None:
        """Retourne le marqueur le plus proche de t dans la tolérance."""
        if not self.markers:
            return None
        best = min(self.markers, key=lambda m: abs(m.time - t))
        return best if abs(best.time - t) <= tol else None

    def prev(self, t: float) -> Marker | None:
        """Marqueur immédiatement avant t (strictement)."""
        candidates = [m for m in self.markers if m.time < t - 1e-6]
        return candidates[-1] if candidates else None

    def next(self, t: float) -> Marker | None:
        """Marqueur immédiatement après t (strictement)."""
        candidates = [m for m in self.markers if m.time > t + 1e-6]
        return candidates[0] if candidates else None

    def clear(self):
        self.markers.clear()

    # ── Sérialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            'markers': [
                {'time': m.time, 'label': m.label, 'color': m.color}
                for m in self.markers
            ]
        }

    def from_dict(self, data: dict):
        self.markers = []
        for md in data.get('markers', []):
            self.add(md['time'], md.get('label', ''), md.get('color', '#F59E0B'))
