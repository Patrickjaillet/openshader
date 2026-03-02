"""
gpu_profiler.py
---------------
v2.1 — Profiler GPU par passe de rendu.

Mesure le temps GPU de chaque pass (Buffer A/B/C/D, Image, Post, Trans)
en utilisant des requêtes OpenGL GL_TIME_ELAPSED via ModernGL.

Sur GPU compatible (OpenGL 3.3+, extension ARB_timer_query),
chaque pass est encadrée par une requête de timer GPU.
Sur GPU incompatible, la mesure CPU est utilisée en fallback.

Usage :
    from .gpu_profiler import GPUProfiler
    profiler = GPUProfiler()
    profiler.initialize(ctx)          # après création du contexte ModernGL

    # Dans la boucle de rendu :
    profiler.begin_pass("Image")
    # ... rendu ...
    profiler.end_pass("Image")

    # Récupération des stats :
    stats = profiler.get_stats()       # dict { pass_name → GPUPassStats }
    profiler.reset()
"""

import time
import collections
from dataclasses import dataclass, field

from .logger import get_logger

log = get_logger(__name__)

# ── Structures ────────────────────────────────────────────────────────────────

@dataclass
class GPUPassStats:
    """Statistiques de timing pour une passe de rendu."""
    name:          str
    last_ms:       float = 0.0     # Temps de la dernière frame (ms)
    avg_ms:        float = 0.0     # Moyenne glissante (N dernières frames)
    min_ms:        float = float('inf')
    max_ms:        float = 0.0
    sample_count:  int   = 0
    _history:      list  = field(default_factory=list, repr=False)

    HISTORY_LEN = 120  # ~2s @ 60fps

    def record(self, ms: float):
        if ms <= 0.0:
            return
        self.last_ms = ms
        self.min_ms  = min(self.min_ms, ms)
        self.max_ms  = max(self.max_ms, ms)
        self._history.append(ms)
        if len(self._history) > self.HISTORY_LEN:
            self._history.pop(0)
        self.avg_ms = sum(self._history) / len(self._history)
        self.sample_count += 1

    def reset(self):
        self.last_ms = 0.0
        self.avg_ms  = 0.0
        self.min_ms  = float('inf')
        self.max_ms  = 0.0
        self.sample_count = 0
        self._history.clear()

    def as_dict(self) -> dict:
        return {
            "name":         self.name,
            "last_ms":      round(self.last_ms,  3),
            "avg_ms":       round(self.avg_ms,   3),
            "min_ms":       round(self.min_ms,   3) if self.min_ms != float('inf') else 0.0,
            "max_ms":       round(self.max_ms,   3),
            "sample_count": self.sample_count,
            "history":      [round(v, 3) for v in self._history[-60:]],
        }


# ── Profiler principal ────────────────────────────────────────────────────────

class GPUProfiler:
    """
    Profiler GPU par passe.

    Modes
    -----
    GPU (GL_TIME_ELAPSED) : précision ≈ 1 µs, disponible sur la plupart des GPU desktop.
    CPU fallback : précision ≈ 1 ms, toujours disponible.

    Le mode GPU nécessite que les requêtes soient récupérées avec un décalage
    d'une frame (latence GPU → CPU). Ce profiler gère automatiquement
    ce décalage via un ring-buffer de requêtes.
    """

    PASS_ORDER = ['Buffer A', 'Buffer B', 'Buffer C', 'Buffer D',
                  'Image', 'Layers', 'Trans', 'Post']

    def __init__(self):
        self._ctx              = None
        self._gpu_mode         = False     # True si GL_TIME_ELAPSED disponible
        self._enabled          = False

        self._stats:   dict[str, GPUPassStats] = {
            p: GPUPassStats(name=p) for p in self.PASS_ORDER
        }

        # Pour le mode CPU fallback
        self._cpu_t0:  dict[str, float] = {}

        # Pour le mode GPU : ring-buffer de requêtes en vol
        # { pass_name → deque[ (query_obj, cpu_fallback_t0) ] }
        self._pending: dict[str, collections.deque] = {
            p: collections.deque(maxlen=4) for p in self.PASS_ORDER
        }

        # Pool de requêtes réutilisables (évite l'allocation par frame)
        self._query_pool:  list = []
        self._pool_cursor: int  = 0

        self._frame_cpu_t0: float = 0.0
        self._total_stats = GPUPassStats(name="Total")

    # ── Init ─────────────────────────────────────────────────────────────────

    def initialize(self, ctx) -> bool:
        """
        Initialise le profiler avec le contexte ModernGL.
        Retourne True si le mode GPU (GL_TIME_ELAPSED) est disponible.
        """
        self._ctx = ctx

        # Vérifie la disponibilité de GL_TIME_ELAPSED
        try:
            q = ctx.query()
            with q:
                pass
            _ = q.elapsed   # lecture test
            self._gpu_mode = True
            # Pré-alloue un pool de requêtes
            self._query_pool = [ctx.query() for _ in range(64)]
            log.info("GPUProfiler : mode GPU (GL_TIME_ELAPSED) activé")
        except Exception as e:
            self._gpu_mode = False
            log.info("GPUProfiler : mode CPU fallback (GL_TIME_ELAPSED indisponible : %s)", e)

        return self._gpu_mode

    # ── Activation ───────────────────────────────────────────────────────────

    def set_enabled(self, enabled: bool):
        """Active ou désactive la collecte de métriques."""
        if enabled != self._enabled:
            self._enabled = enabled
            if not enabled:
                self._flush_pending()
            log.debug("GPUProfiler : %s", "activé" if enabled else "désactivé")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def gpu_mode(self) -> bool:
        return self._gpu_mode

    # ── Mesure ───────────────────────────────────────────────────────────────

    def begin_frame(self):
        """Appelé une fois par frame avant les passes."""
        if not self._enabled:
            return
        self._frame_cpu_t0 = time.perf_counter()
        # Récupère les résultats des requêtes GPU en attente de la frame précédente
        if self._gpu_mode:
            self._collect_pending()

    def end_frame(self):
        """Appelé une fois par frame après toutes les passes."""
        if not self._enabled:
            return
        elapsed_ms = (time.perf_counter() - self._frame_cpu_t0) * 1000.0
        self._total_stats.record(elapsed_ms)

    def begin_pass(self, pass_name: str):
        """Démarre la mesure de la passe *pass_name*."""
        if not self._enabled or pass_name not in self._stats:
            return

        if self._gpu_mode:
            q = self._get_query()
            if q:
                try:
                    q.__enter__()
                    self._cpu_t0[pass_name] = time.perf_counter()
                    # On stocke la requête pour end_pass
                    self._cpu_t0[f'_q_{pass_name}'] = q
                except Exception:
                    pass
        else:
            self._cpu_t0[pass_name] = time.perf_counter()

    def end_pass(self, pass_name: str):
        """Termine la mesure de la passe *pass_name*."""
        if not self._enabled or pass_name not in self._stats:
            return

        if self._gpu_mode:
            q = self._cpu_t0.pop(f'_q_{pass_name}', None)
            cpu_t0 = self._cpu_t0.pop(pass_name, None)
            if q:
                try:
                    q.__exit__(None, None, None)
                    # La requête GPU n'est pas encore disponible — on la file
                    self._pending[pass_name].append((q, cpu_t0 or time.perf_counter()))
                    self._recycle_query(q)  # le résultat est déjà dans la queue avant recycle
                except Exception:
                    pass
        else:
            t0 = self._cpu_t0.pop(pass_name, None)
            if t0 is not None:
                ms = (time.perf_counter() - t0) * 1000.0
                self._stats[pass_name].record(ms)

    # ── Lecture des résultats ─────────────────────────────────────────────────

    def _collect_pending(self):
        """
        Récupère les résultats GPU disponibles pour les requêtes en attente.
        Appelé au début de chaque frame (décalage d'une frame).
        """
        for pass_name, queue in self._pending.items():
            if not queue:
                continue
            q, _cpu_t0 = queue[0]
            try:
                ns = q.elapsed   # nanosecondes ; disponible 1-2 frames plus tard
                if ns > 0:
                    ms = ns / 1_000_000.0
                    self._stats[pass_name].record(ms)
                    queue.popleft()
            except Exception:
                # Pas encore disponible — on réessaiera la prochaine frame
                pass

    def _flush_pending(self):
        """Vide la queue des requêtes en attente (lors de la désactivation)."""
        for queue in self._pending.values():
            queue.clear()

    # ── Pool de requêtes ──────────────────────────────────────────────────────

    def _get_query(self):
        """Retourne une requête du pool (round-robin)."""
        if not self._query_pool:
            return None
        q = self._query_pool[self._pool_cursor % len(self._query_pool)]
        self._pool_cursor += 1
        return q

    def _recycle_query(self, q):
        """Remet la requête dans le pool (no-op — le pool est statique)."""
        pass

    # ── Statistiques ─────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, GPUPassStats]:
        """Retourne le dict complet des statistiques par passe."""
        return dict(self._stats)

    def get_total(self) -> GPUPassStats:
        """Retourne les stats CPU du frame total."""
        return self._total_stats

    def get_summary(self) -> dict:
        """
        Retourne un résumé sérialisable (pour export JSON / affichage UI).
        """
        result = {
            "gpu_mode": self._gpu_mode,
            "enabled":  self._enabled,
            "passes":   {p: s.as_dict() for p, s in self._stats.items()
                         if s.sample_count > 0},
            "total":    self._total_stats.as_dict(),
        }
        return result

    def reset(self):
        """Remet à zéro toutes les statistiques."""
        for s in self._stats.values():
            s.reset()
        self._total_stats.reset()
        self._flush_pending()
        log.debug("GPUProfiler : stats réinitialisées")

    def format_overlay(self) -> str:
        """
        Retourne une chaîne multi-ligne prête pour l'affichage en overlay.
        Format :  PASS         avg_ms   last_ms   max_ms
        """
        if not self._enabled:
            return "Profiler GPU désactivé"
        lines = [f"{'PASSE':<14} {'avg':>6} {'last':>6} {'max':>6}",
                 "─" * 36]
        total_avg = 0.0
        for pass_name in self.PASS_ORDER:
            s = self._stats[pass_name]
            if s.sample_count == 0:
                continue
            lines.append(
                f"{pass_name:<14} {s.avg_ms:>5.2f}ms {s.last_ms:>5.2f}ms {s.max_ms:>5.2f}ms"
            )
            total_avg += s.avg_ms
        lines.append("─" * 36)
        lines.append(f"{'Total CPU':<14} {self._total_stats.avg_ms:>5.2f}ms")
        mode = "GPU" if self._gpu_mode else "CPU"
        lines.append(f"Mode : {mode}")
        return "\n".join(lines)
