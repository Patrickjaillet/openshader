"""
timeline.py
-----------
Modèle de données pour la timeline, les pistes (tracks) et les keyframes.
Gère l'interpolation des valeurs entre keyframes.

v2.2 — Nouvelles fonctionnalités :
  - Keyframe.expression : expression Python sandboxée ("sin(t*2)*0.5+0.5")
    Variables disponibles : t, beat, bpm, rms, fft_n
  - Track.value_type == 'camera' : piste de caméra 3D composite
    Injecte uCamPos, uCamTarget, uCamFOV à chaque frame
  - Track.group / Track.group_folded : groupement visuel de pistes
  - Timeline.add_camera_track() : API de création de piste caméra

v1.2 — Ajout de l'interpolation Bézier :
  - BezierHandle(dt, dv) en coordonnées relatives au keyframe parent
  - Keyframe.handle_in / handle_out (non-breaking : ignorés si interp != 'bezier')
  - Track.apply_auto_tangents() pour le mode Catmull-Rom automatique
  - Sérialisation rétrocompatible (.dmk v1.1.x se chargent sans modification)
"""

from __future__ import annotations
import json
import bisect
import math
import random
from dataclasses import dataclass, field
from typing import Any

from .bezier  import bezier_interpolate, catmull_rom_tangent, auto_tangent_endpoint
from .marker  import MarkerTrack


# ── Sandbox d'expression ─────────────────────────────────────────────────────

_EXPR_SAFE_GLOBALS: dict = {
    "__builtins__": {},
    "sin":   math.sin,   "cos":   math.cos,   "tan":   math.tan,
    "asin":  math.asin,  "acos":  math.acos,  "atan":  math.atan,
    "atan2": math.atan2, "sqrt":  math.sqrt,  "abs":   abs,
    "floor": math.floor, "ceil":  math.ceil,  "round": round,
    "min":   min,        "max":   max,         "pow":   math.pow,
    "exp":   math.exp,   "log":   math.log,   "pi":    math.pi,
    "tau":   math.tau,   "e":     math.e,
    "fract": lambda x: x - math.floor(x),
    "clamp": lambda x, lo, hi: max(lo, min(hi, x)),
    "mix":   lambda a, b, t: a + (b - a) * t,
    "step":  lambda edge, x: 0.0 if x < edge else 1.0,
    "smoothstep": lambda e0, e1, x: (lambda t: t*t*(3-2*t))(max(0.0, min(1.0, (x-e0)/(e1-e0) if e1!=e0 else 0.0))),
    "random": random.random,
}

def _eval_expression(expr: str, t: float, bpm: float = 120.0,
                     rms: float = 0.0, fft: tuple = ()) -> Any:
    """
    Évalue une expression Python sandboxée.
    Variables disponibles : t, beat, bpm, rms, fft[n]
    Retourne le résultat (float) ou None en cas d'erreur.
    """
    beat = t * bpm / 60.0

    class _FFTProxy:
        def __getitem__(self, n):
            return fft[n] if n < len(fft) else 0.0

    local_vars = {
        "t":    t,
        "beat": beat,
        "bpm":  bpm,
        "rms":  rms,
        "fft":  _FFTProxy(),
    }
    try:
        return float(eval(expr, _EXPR_SAFE_GLOBALS, local_vars))  # noqa: S307
    except Exception:
        return None

@dataclass
class BezierHandle:
    """
    Handle tangent d'un keyframe Bézier.
    Coordonnées relatives au keyframe parent :
      - dt > 0  →  handle vers le futur   (handle_out)
      - dt < 0  →  handle vers le passé   (handle_in)
      - dv      →  delta de valeur
    """
    dt: float = 0.0
    dv: Any   = 0.0   # float pour les scalaires, tuple pour les vecs


@dataclass(eq=False)
class Keyframe:
    """Une image-clé : temps + valeur + type d'interpolation."""
    time:   float            # en secondes
    value:  Any              # float, tuple, etc.
    interp: str = 'linear'   # 'linear' | 'step' | 'smooth' | 'bezier'

    # Handles Bézier — ignorés si interp != 'bezier'
    handle_in:  BezierHandle = field(default_factory=BezierHandle)
    handle_out: BezierHandle = field(default_factory=BezierHandle)

    # Indique si les handles sont liés (tangente continue) ou brisés (indépendants)
    handles_linked: bool = True

    # v2.2 — Expression Python optionnelle (remplace la valeur fixe si non vide)
    # Ex: "sin(t * 2.0) * 0.5 + 0.5"   Variables: t, beat, bpm, rms, fft[n]
    expression: str = ""


@dataclass(eq=False)
class Track:
    """
    Une piste contrôlant un uniform shader.
    Ex: uniform 'uIntensity' (float), ou 'uColor' (vec3).
    v2.2 : value_type 'camera' pour pistes de caméra 3D.
    """
    name:         str                 # Nom lisible (ex: "Intensité")
    uniform_name: str                 # Nom GLSL (ex: "uIntensity")
    value_type:   str = 'float'       # 'float' | 'vec2' | 'vec3' | 'vec4' | 'shader' | 'audio' | 'camera'
    keyframes:    list[Keyframe] = field(default_factory=list)
    color:        str  = "#22252e"    # Couleur de fond de la piste (hex)
    enabled:      bool = True
    audio_path:   str  = ""           # Pour value_type=='audio' : chemin du fichier
    height:       int  = 38           # Hauteur visuelle en pixels (zoom vertical v1.2)

    # v2.2 — Groupement de pistes
    group:        str  = ""           # Nom du groupe (vide = pas de groupe)
    group_folded: bool = False        # Replié ou déplié

    # ── Gestion des keyframes ───────────────────────────────────────────────

    def add_keyframe(self, t: float, value: Any, interp: str = 'linear') -> Keyframe:
        """Ajoute ou remplace un keyframe à l'instant t."""
        # Remplace si un KF existe à ±0.01s
        for i, kf in enumerate(self.keyframes):
            if abs(kf.time - t) < 0.01:
                # Conserve les handles Bézier existants si le mode ne change pas
                old_in  = kf.handle_in
                old_out = kf.handle_out
                self.keyframes[i] = Keyframe(t, value, interp,
                                             handle_in=old_in, handle_out=old_out)
                return self.keyframes[i]
        kf = Keyframe(t, value, interp)
        idx = bisect.bisect_left([k.time for k in self.keyframes], t)
        self.keyframes.insert(idx, kf)
        return kf

    def remove_keyframe(self, t: float):
        """Supprime le keyframe le plus proche de t."""
        if not self.keyframes:
            return
        idx = min(range(len(self.keyframes)),
                  key=lambda i: abs(self.keyframes[i].time - t))
        self.keyframes.pop(idx)

    def move_keyframe(self, old_t: float, new_t: float):
        """Déplace un keyframe d'un temps à un autre (conserve ses handles)."""
        for kf in self.keyframes:
            if abs(kf.time - old_t) < 0.01:
                kf.time = new_t
                self.keyframes.sort(key=lambda k: k.time)
                return

    def get_value_at(self, t: float, bpm: float = 120.0,
                     rms: float = 0.0, fft: tuple = ()) -> Any:
        """
        Retourne la valeur interpolée à l'instant t.
        Si un keyframe a une expression définie, elle prend le dessus sur la valeur fixe.
        Si aucun KF : retourne None.
        Si 1 seul KF : retourne sa valeur (ou expression).
        """
        if not self.keyframes:
            return None

        # Cas 1 KF ou avant le premier KF
        if len(self.keyframes) == 1 or t <= self.keyframes[0].time:
            kf = self.keyframes[0]
            if kf.expression:
                result = _eval_expression(kf.expression, t, bpm, rms, fft)
                if result is not None:
                    return result
            return kf.value

        # Après le dernier KF
        if t >= self.keyframes[-1].time:
            kf = self.keyframes[-1]
            if kf.expression:
                result = _eval_expression(kf.expression, t, bpm, rms, fft)
                if result is not None:
                    return result
            return kf.value

        # Trouve les deux KFs encadrants
        times = [kf.time for kf in self.keyframes]
        idx   = bisect.bisect_right(times, t) - 1
        kf_a  = self.keyframes[idx]
        kf_b  = self.keyframes[idx + 1]

        # Si le KF entrant a une expression, on l'évalue pour t courant
        if kf_b.expression:
            result = _eval_expression(kf_b.expression, t, bpm, rms, fft)
            if result is not None:
                return result

        # Paramètre d'interpolation [0, 1] pour les modes non-Bézier
        span  = kf_b.time - kf_a.time
        alpha = (t - kf_a.time) / span if span > 0 else 0.0

        # Bézier : utilise les handles des deux KFs encadrants
        if kf_b.interp == 'bezier':
            return bezier_interpolate(
                kf_a.time, kf_a.value,
                kf_a.handle_out.dt, kf_a.handle_out.dv,
                kf_b.time, kf_b.value,
                kf_b.handle_in.dt,  kf_b.handle_in.dv,
                t,
            )

        return _interpolate(kf_a.value, kf_b.value, alpha, kf_b.interp)

    def get_default_value(self) -> Any:
        """Valeur par défaut selon le type."""
        defaults = {
            'float':  1.0,
            'vec2':   (0.0, 0.0),
            'vec3':   (1.0, 1.0, 1.0),
            'vec4':   (1.0, 1.0, 1.0, 1.0),
            'shader': '',
            # camera : (pos_x, pos_y, pos_z, tgt_x, tgt_y, tgt_z, fov)
            'camera': (0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 45.0),
        }
        return defaults.get(self.value_type, 1.0)

    # ── Tangentes automatiques ──────────────────────────────────────────────

    def apply_auto_tangents(self, kf: Keyframe | None = None):
        """
        Calcule et applique les handles Catmull-Rom sur un keyframe donné
        (ou sur tous les KFs de la piste si kf=None).

        N'affecte que les keyframes dont interp == 'bezier'.
        Les handles existants sont écrasés.
        """
        kfs = self.keyframes
        targets = [kf] if kf else kfs

        for i, k in enumerate(kfs):
            if k not in targets or k.interp != 'bezier':
                continue

            if len(kfs) == 1:
                # Un seul KF : handles nuls
                k.handle_in  = BezierHandle(0.0, _zero_dv(k.value))
                k.handle_out = BezierHandle(0.0, _zero_dv(k.value))

            elif i == 0:
                # Premier KF : tangente vers le suivant
                out_dt, out_dv = auto_tangent_endpoint(
                    k.time, k.value, kfs[i + 1].time, kfs[i + 1].value, 'out')
                k.handle_in  = BezierHandle(0.0, _zero_dv(k.value))
                k.handle_out = BezierHandle(out_dt, out_dv)

            elif i == len(kfs) - 1:
                # Dernier KF : tangente vers le précédent
                in_dt, in_dv = auto_tangent_endpoint(
                    k.time, k.value, kfs[i - 1].time, kfs[i - 1].value, 'in')
                k.handle_in  = BezierHandle(in_dt, in_dv)
                k.handle_out = BezierHandle(0.0, _zero_dv(k.value))

            else:
                # KF intermédiaire : Catmull-Rom
                in_dt, in_dv, out_dt, out_dv = catmull_rom_tangent(
                    kfs[i - 1].time, kfs[i - 1].value,
                    k.time,          k.value,
                    kfs[i + 1].time, kfs[i + 1].value,
                )
                k.handle_in  = BezierHandle(in_dt,  in_dv)
                k.handle_out = BezierHandle(out_dt, out_dv)


def _zero_dv(value):
    """Retourne un delta-valeur nul du bon type."""
    if isinstance(value, (int, float)):
        return 0.0
    if isinstance(value, (tuple, list)):
        return tuple(0.0 for _ in value)
    return 0.0


def _interpolate(a, b, t: float, interp: str):
    """Interpole entre deux valeurs selon le mode (hors Bézier)."""
    if not isinstance(a, (int, float, tuple, list)):
        return a

    if interp == 'step':
        return a
    if interp == 'smooth':
        t = t * t * (3 - 2 * t)  # smoothstep

    if isinstance(a, (int, float)):
        return a + (b - a) * t

    if isinstance(a, (tuple, list)):
        return tuple(av + (bv - av) * t for av, bv in zip(a, b))

    return a


# ── Timeline ─────────────────────────────────────────────────────────────────

class Timeline:
    """
    Gestionnaire principal de la timeline.
    Contient plusieurs Track et calcule les uniforms à un instant donné.
    """

    def __init__(self, duration: float = 30.0):
        self.duration:       float = duration
        self.tracks:         list[Track] = []
        # v6.0 — Tempo map (BPM automation) — None until Arrangement View is used
        self.tempo_map_data: dict = {}   # sérialisé si non vide
        self._current_time:  float = 0.0
        self.marker_track:   MarkerTrack = MarkerTrack()
        # Snap BPM
        self.bpm:            float = 120.0
        self.snap_to_grid:   bool  = False
        self.snap_division:  int   = 4   # 1=mesure, 2=demi, 4=quart, 8=huitième
        # Loop region
        self.loop_enabled:   bool  = False
        self.loop_in:        float = 0.0
        self.loop_out:       float = duration

    # ── Pistes ──────────────────────────────────────────────────────────────

    def add_track(self, name: str, uniform_name: str,
                  value_type: str = 'float') -> Track:
        track = Track(name=name, uniform_name=uniform_name, value_type=value_type)
        self.tracks.append(track)
        return track

    def add_audio_track(self, name: str, audio_path: str) -> Track:
        track = Track(name=name, uniform_name='_audio', value_type='audio',
                      color='#1a2a1a', audio_path=audio_path)
        self.tracks.append(track)
        return track

    def add_camera_track(self, name: str = "Caméra") -> Track:
        """
        v2.2 — Crée une piste de caméra 3D.
        La valeur est un tuple (px, py, pz, tx, ty, tz, fov) représentant
        position, cible et champ de vision.
        Uniforms injectés : uCamPos (vec3), uCamTarget (vec3), uCamFOV (float).
        """
        track = Track(
            name=name,
            uniform_name='_camera',
            value_type='camera',
            color='#1a1a3a',
            height=50,
        )
        self.tracks.append(track)
        return track

    def remove_track(self, track: Track):
        if track in self.tracks:
            self.tracks.remove(track)

    def get_track_by_uniform(self, uniform_name: str) -> Track | None:
        for t in self.tracks:
            if t.uniform_name == uniform_name:
                return t
        return None

    def clear(self):
        self.tracks = []
        self.duration = 60.0
        self.marker_track.clear()

    # ── Snap BPM ────────────────────────────────────────────────────────────

    def beat_to_time(self, beat: float) -> float:
        """Convertit un numéro de beat en secondes."""
        return beat * 60.0 / max(1e-6, self.bpm)

    def time_to_beat(self, t: float) -> float:
        """Convertit des secondes en numéro de beat."""
        return t * self.bpm / 60.0

    def snap(self, t: float) -> float:
        """
        Arrondit t à la subdivision BPM active.
        No-op si snap_to_grid est False.
        """
        if not self.snap_to_grid or self.bpm <= 0:
            return t
        grid = 60.0 / self.bpm / max(1, self.snap_division)
        return round(t / grid) * grid

    # ── Évaluation ──────────────────────────────────────────────────────────

    def evaluate(self, t: float, rms: float = 0.0, fft: tuple = ()) -> dict[str, Any]:
        """
        Retourne un dict {uniform_name: value} pour toutes les pistes actives
        à l'instant t.
        Les pistes de type 'camera' injectent uCamPos / uCamTarget / uCamFOV.
        Les pistes avec expressions en keyframes les évaluent en sandboxe.
        """
        self._current_time = t
        result = {}
        for track in self.tracks:
            if not track.enabled:
                continue
            val = track.get_value_at(t, bpm=self.bpm, rms=rms, fft=fft)
            if val is None:
                val = track.get_default_value()

            if track.value_type == 'camera':
                # Décompose le tuple (px, py, pz, tx, ty, tz, fov)
                if isinstance(val, (tuple, list)) and len(val) >= 7:
                    result['uCamPos']    = (val[0], val[1], val[2])
                    result['uCamTarget'] = (val[3], val[4], val[5])
                    result['uCamFOV']    = float(val[6])
                else:
                    result['uCamPos']    = (0.0, 0.0, 3.0)
                    result['uCamTarget'] = (0.0, 0.0, 0.0)
                    result['uCamFOV']    = 45.0
            else:
                result[track.uniform_name] = val
        return result

    # ── Sérialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        def _handle_to_dict(h: BezierHandle) -> dict:
            dv = list(h.dv) if isinstance(h.dv, tuple) else h.dv
            return {'dt': h.dt, 'dv': dv}

        return {
            'duration':       self.duration,
            'tempo_map':      self.tempo_map_data,   # v6.0
            'markers':        self.marker_track.to_dict(),
            'bpm':            self.bpm,
            'snap_to_grid':   self.snap_to_grid,
            'snap_division':  self.snap_division,
            'loop_enabled':   self.loop_enabled,
            'loop_in':        self.loop_in,
            'loop_out':       self.loop_out,
            'tracks': [
                {
                    'name':         t.name,
                    'height':       t.height,
                    'uniform_name': t.uniform_name,
                    'value_type':   t.value_type,
                    'color':        t.color,
                    'enabled':      t.enabled,
                    'audio_path':   t.audio_path,
                    # v2.2 — groupe
                    'group':        t.group,
                    'group_folded': t.group_folded,
                    'keyframes': [
                        {
                            'time':           kf.time,
                            'value':          list(kf.value) if isinstance(kf.value, tuple) else kf.value,
                            'interp':         kf.interp,
                            # v2.2 — expression
                            **({'expression': kf.expression} if kf.expression else {}),
                            # Handles Bézier — omis si nuls pour garder les fichiers légers
                            **({'handle_in':  _handle_to_dict(kf.handle_in),
                                'handle_out': _handle_to_dict(kf.handle_out),
                                'handles_linked': kf.handles_linked}
                               if kf.interp == 'bezier' else {}),
                        }
                        for kf in t.keyframes
                    ]
                }
                for t in self.tracks
            ]
        }

    def from_dict(self, data: dict):
        self.duration      = data.get('duration', 30.0)
        self.tempo_map_data = data.get('tempo_map', {})   # v6.0
        self.marker_track.from_dict(data.get('markers', {}))
        self.bpm           = data.get('bpm', 120.0)
        self.snap_to_grid  = data.get('snap_to_grid', False)
        self.snap_division = data.get('snap_division', 4)
        self.loop_enabled  = data.get('loop_enabled', False)
        self.loop_in       = data.get('loop_in', 0.0)
        self.loop_out      = data.get('loop_out', self.duration)
        self.tracks        = []

        for td in data.get('tracks', []):
            track = Track(
                name=td['name'],
                uniform_name=td['uniform_name'],
                value_type=td.get('value_type', 'float'),
                color=td.get('color', "#22252e"),
                enabled=td.get('enabled', True),
                audio_path=td.get('audio_path', ''),
                height=td.get('height', 38),
                group=td.get('group', ''),
                group_folded=td.get('group_folded', False),
            )
            for kfd in td.get('keyframes', []):
                val = kfd['value']
                if isinstance(val, list):
                    val = tuple(val)

                kf = track.add_keyframe(kfd['time'], val, kfd.get('interp', 'linear'))

                # v2.2 — restaure expression
                kf.expression = kfd.get('expression', '')

                # Restaure les handles Bézier s'ils sont présents
                if 'handle_in' in kfd:
                    h = kfd['handle_in']
                    dv = tuple(h['dv']) if isinstance(h['dv'], list) else h['dv']
                    kf.handle_in = BezierHandle(h['dt'], dv)
                if 'handle_out' in kfd:
                    h = kfd['handle_out']
                    dv = tuple(h['dv']) if isinstance(h['dv'], list) else h['dv']
                    kf.handle_out = BezierHandle(h['dt'], dv)
                kf.handles_linked = kfd.get('handles_linked', True)

            self.tracks.append(track)

    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def load(self, filepath: str):
        """Charge la timeline depuis un fichier JSON. Lève ValueError si corrompu."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Fichier timeline corrompu ({filepath}): {e}") from e
        except (OSError, KeyError, TypeError) as e:
            raise ValueError(f"Impossible de charger la timeline ({filepath}): {e}") from e
