"""Sliding-window regime tracking (v0.6).

Generic change detection on the two classification axes: classify K over
rolling time windows and flag windows where the coherence class changes
(e.g. broadband/monotone -> recurrent coherent mode). Domain-agnostic.

Honest caveats: the transport exponent inside a short window is indicative
only (fitted on the window's absolute times); the reliable change signal is
the coherence axis. A window must span >= 2 oscillation periods to call
RING_RECURRENT.
"""
import numpy as np
from ..regimes.classifier import classify_any
from ..core.kernel import Kernel
from ..core.kernel2d import Kernel2D
from ..core.kernel3d import Kernel3D
from ..core.kernel4d import Kernel4D


def _time_slice(kernel, sl):
    if isinstance(kernel, Kernel4D):
        return Kernel4D(kernel.x, kernel.y, kernel.z, kernel.w, kernel.t[sl], kernel.K[sl])
    if isinstance(kernel, Kernel3D):
        return Kernel3D(kernel.x, kernel.y, kernel.z, kernel.t[sl], kernel.K[sl])
    if isinstance(kernel, Kernel2D):
        return Kernel2D(kernel.x, kernel.y, kernel.t[sl], kernel.K[sl])
    return Kernel(kernel.r, kernel.t[sl], kernel.K[sl])


def track_regimes(kernel, window, stride=1, **classify_kw):
    """Classify sliding windows of `window` frames every `stride` frames.

    Returns a list of dicts with t_start, t_end, regime, coherence, exponent,
    and transition (True where the coherence class differs from the previous
    window)."""
    if window < 4 or window > len(kernel.t):
        raise ValueError("window must be in [4, T]")
    out = []
    prev = None
    for s in range(0, len(kernel.t) - window + 1, stride):
        sub = _time_slice(kernel, slice(s, s + window))
        rep = classify_any(sub, **classify_kw)
        out.append({
            "t_start": float(sub.t[0]), "t_end": float(sub.t[-1]),
            "regime": rep.regime.value, "coherence": rep.coherence,
            "exponent": float(rep.exponent) if np.isfinite(rep.exponent) else None,
            "transition": prev is not None and rep.coherence != prev,
        })
        prev = rep.coherence
    return out
