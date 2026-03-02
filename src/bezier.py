"""
bezier.py
---------
Math Bézier cubique pour l'interpolation de keyframes.
Isolé de la timeline pour faciliter les tests unitaires.
"""

from __future__ import annotations
from typing import Union

Scalar = Union[int, float]


# ── Évaluation ──────────────────────────────────────────────────────────────

def cubic_bezier(p0: Scalar, p1: Scalar, p2: Scalar, p3: Scalar, t: float) -> float:
    """
    Évalue un Bézier cubique entre p0 et p3 avec les tangentes p1 et p2.
    t ∈ [0, 1]
    """
    mt = 1.0 - t
    return mt**3 * p0 + 3.0 * mt**2 * t * p1 + 3.0 * mt * t**2 * p2 + t**3 * p3


def cubic_bezier_tuple(
    p0: tuple, p1: tuple, p2: tuple, p3: tuple, t: float
) -> tuple:
    """Version vectorielle : évalue composante par composante."""
    return tuple(
        cubic_bezier(a, b, c, d, t)
        for a, b, c, d in zip(p0, p1, p2, p3)
    )


# ── Résolution inverse ───────────────────────────────────────────────────────

def solve_t_for_x(
    x0: float, x1: float, x2: float, x3: float,
    target_x: float,
    tol: float = 1e-4,
    max_iter: int = 32,
) -> float:
    """
    Résolution numérique par dichotomie :
    trouve t ∈ [0,1] tel que cubic_bezier(x0, x1, x2, x3, t) ≈ target_x.

    Précision ±tol, suffisant pour 60 fps.
    Retourne 0.0 si target_x < x0, 1.0 si target_x > x3.
    """
    if target_x <= x0:
        return 0.0
    if target_x >= x3:
        return 1.0

    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = (lo + hi) * 0.5
        val = cubic_bezier(x0, x1, x2, x3, mid)
        if abs(val - target_x) < tol:
            return mid
        if val < target_x:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


# ── Tangentes automatiques (Catmull-Rom) ─────────────────────────────────────

def catmull_rom_tangent(
    t_prev: float, v_prev,
    t_curr: float, v_curr,
    t_next: float, v_next,
) -> tuple:
    """
    Calcule les deux handles Bézier (in, out) pour un keyframe intermédiaire
    en utilisant l'algorithme Catmull-Rom.

    Retourne (handle_in_dt, handle_in_dv, handle_out_dt, handle_out_dv)
    en coordonnées relatives au keyframe courant.
    """
    span_prev = max(1e-6, t_curr - t_prev)
    span_next = max(1e-6, t_next - t_curr)

    def _tangent(a, b, span):
        if isinstance(a, (int, float)):
            return (b - a) / span
        return tuple((bv - av) / span for av, bv in zip(a, b))

    def _scale(v, factor):
        if isinstance(v, (int, float)):
            return v * factor
        return tuple(x * factor for x in v)

    # Tangente Catmull-Rom au point courant
    tan = _tangent(v_prev, v_next, t_next - t_prev)

    # Handle out : 1/3 de la distance au suivant
    out_dt = span_next / 3.0
    out_dv = _scale(tan, out_dt)

    # Handle in : -1/3 de la distance au précédent
    in_dt  = -span_prev / 3.0
    in_dv  = _scale(tan, in_dt)

    return in_dt, in_dv, out_dt, out_dv


def auto_tangent_endpoint(
    t_curr: float, v_curr,
    t_neighbor: float, v_neighbor,
    side: str,  # 'out' ou 'in'
) -> tuple:
    """
    Calcule une tangente automatique pour un keyframe de début ou de fin.
    side='out' → premier KF,  side='in' → dernier KF.
    """
    span = max(1e-6, abs(t_neighbor - t_curr))
    handle_dt = (span / 3.0) * (1 if side == 'out' else -1)

    if isinstance(v_curr, (int, float)):
        slope = (v_neighbor - v_curr) / span
        handle_dv = slope * abs(handle_dt)
    else:
        handle_dv = tuple(
            ((bv - av) / span) * abs(handle_dt)
            for av, bv in zip(v_curr, v_neighbor)
        )

    return handle_dt, handle_dv


# ── Interpolation complète entre deux KFs Bézier ────────────────────────────

def bezier_interpolate(
    t_a: float, v_a, h_out_dt: float, h_out_dv,
    t_b: float, v_b, h_in_dt: float,  h_in_dv,
    t: float,
):
    """
    Interpole la valeur à l'instant t entre deux keyframes A et B
    en utilisant leurs handles Bézier.

    h_out_dt / h_out_dv : handle de sortie du KF A (en coordonnées relatives à A)
    h_in_dt  / h_in_dv  : handle d'entrée du KF B (en coordonnées relatives à B)
    """
    span = max(1e-9, t_b - t_a)

    # Positions absolues des points de contrôle (temps)
    cx1 = t_a + h_out_dt
    cx2 = t_b + h_in_dt

    # Résolution numérique : trouve u ∈ [0,1] tel que bezier_time(u) == t
    u = solve_t_for_x(t_a, cx1, cx2, t_b, t)

    # Interpolation de la valeur
    def _ctrl(base, delta):
        if isinstance(base, (int, float)):
            return base + delta
        return tuple(bv + dv for bv, dv in zip(base, delta))

    cy1 = _ctrl(v_a, h_out_dv)
    cy2 = _ctrl(v_b, h_in_dv)

    if isinstance(v_a, (int, float)):
        return cubic_bezier(v_a, cy1, cy2, v_b, u)
    return cubic_bezier_tuple(v_a, cy1, cy2, v_b, u)
