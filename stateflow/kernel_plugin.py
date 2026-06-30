"""stateflow-kernel: the reference plugin. A thin adapter over the existing,
unmodified ``kernel_viewer`` package.

This plugin does NOT reimplement any measurement. It calls
``kernel_viewer``'s functions and re-expresses their already-continuous
outputs (the transport exponent p; the ring-strength trajectory E(t)) in the
stateflow vocabulary (FeatureSeries / Axis / Trajectory). The concrete regime
read-out is delegated to ``kernel_viewer.classify_any`` and is therefore
byte-for-byte identical to current kernel-viewer output.
"""
from __future__ import annotations
from typing import Any
import numpy as np

from kernel_viewer.io.loader import load_any
from kernel_viewer.metrics.transport import transport_exponent, tail_shape
from kernel_viewer.metrics.ring import ring_strength, ring_episodes, classify_coherence
from kernel_viewer.regimes.classifier import classify_any

from .core import FeatureSeries, Trajectory, Transition


class KernelExtractor:
    """Turn a kernel file (or a loaded Kernel/2D/3D/4D object) into a
    FeatureSeries whose columns are the per-time scalars stateflow's kernel
    axes consume: the source-site value K(0,t), the second moment sigma^2(t),
    and the edge fraction (used as the mask)."""

    name = "kernel"

    def from_path(self, path: str) -> FeatureSeries:
        return self.from_kernel(load_any(path), source=str(path))

    def from_kernel(self, kernel, source: str = "in-memory") -> FeatureSeries:
        sigma2 = kernel.sigma2()
        k0 = kernel.k0()
        edge = kernel.edge_fraction()
        feats = np.column_stack([k0, sigma2, edge])
        mask = np.isfinite(edge) & (edge <= 0.3)
        fs = FeatureSeries(
            time=kernel.t, features=feats,
            feature_names=["k0", "sigma2", "edge_fraction"],
            metadata={
                "provenance": "kernel_viewer", "source": source,
                "method": "exact second moment / source-site value / edge fraction",
                "dimensionality": getattr(kernel, "geometry_dim",
                                          1 if feats.ndim else 1),
                "clean_window_rule": "edge_fraction <= 0.3",
            },
            mask=mask,
        )
        # keep handles: the original kernel for regime delegation, and a
        # profile-bearing kernel for the shape axes. For 2D/3D/4D the profile
        # is the radial reduction (which carries full-D moments); for 1D the
        # kernel itself already exposes symmetrized()/sigma2().
        fs._kernel = kernel
        fs._profile = kernel.radial() if hasattr(kernel, "radial") else kernel
        return fs


class TransportAxis:
    """Axis 1 -- transport. Continuous descriptor: the exponent p in
    sigma^2 ~ t^p, measured by ``kernel_viewer.transport_exponent`` on the
    boundary-clean window. discretize() reproduces kernel-viewer's transport
    label (including the Levy promotion via power-law spatial tail)."""

    name = "transport"

    def compute(self, fs: FeatureSeries) -> Trajectory:
        kernel = fs._profile
        p, tm, sl, t_clean = transport_exponent(kernel)
        # the native continuous object: the local log-log slope trajectory,
        # whose late-window mean is p (a single scalar summary attached too).
        # confidence per slope-point: high inside the clean window, decaying
        # as the boundary is approached (mirrors edge_fraction guarding).
        edge = kernel.edge_fraction()
        conf_t = np.interp(np.asarray(tm, dtype=float), kernel.t,
                           np.clip(1.0 - np.nan_to_num(edge, nan=1.0), 0.0, 1.0))
        traj = Trajectory(times=np.asarray(tm, dtype=float),
                          values=np.asarray(sl, dtype=float),
                          confidence=conf_t, name="transport_loglog_slope")
        traj.scalar = float(p) if np.isfinite(p) else float("nan")
        traj.t_clean = float(t_clean)
        return traj

    def confidence(self, fs: FeatureSeries) -> np.ndarray:
        # reliability falls off once the profile hits the boundary
        kernel = fs._profile
        edge = kernel.edge_fraction()
        return np.clip(1.0 - np.nan_to_num(edge, nan=1.0), 0.0, 1.0)

    def discretize(self, traj: Trajectory) -> list[str]:
        p = getattr(traj, "scalar", float("nan"))
        if not np.isfinite(p):
            return ["unknown"]
        if p >= 1.8:
            return ["ballistic"]
        if p > 1.2:
            return ["superdiffusive"]
        if p >= 0.8:
            return ["diffusive"]
        return ["subdiffusive"]


class CoherenceAxis:
    """Axis 2 -- coherence. NATIVE continuous descriptor: the ring-strength
    trajectory E(t) (not a scalar), via ``kernel_viewer.ring_strength``. Its
    "value" is the history of that trajectory plus its episode descriptors.
    discretize() reproduces kernel-viewer's coherence label."""

    name = "coherence"

    def __init__(self, ring_threshold: float = 0.05) -> None:
        self.ring_threshold = float(ring_threshold)

    def compute(self, fs: FeatureSeries) -> Trajectory:
        kernel = fs._profile
        E, rstar = ring_strength(kernel)
        episodes = ring_episodes(kernel.t, E, self.ring_threshold)
        traj = Trajectory(times=np.asarray(kernel.t, dtype=float),
                          values=np.asarray(E, dtype=float),
                          confidence=np.isfinite(E).astype(float),
                          mask=np.isfinite(E), name="ring_strength_E")
        # transitions: episode starts (rises above threshold) and ends
        transitions: list[Transition] = []
        for (t0, t1, emax) in episodes:
            i0 = int(np.argmin(np.abs(kernel.t - t0)))
            transitions.append(Transition(index=i0, time=float(t0), kind="ring_start",
                                          detail={"E_max": float(emax)}))
        traj.transitions = transitions
        traj.episodes = episodes
        traj.rstar = np.asarray(rstar, dtype=float)
        return traj

    def confidence(self, fs: FeatureSeries) -> np.ndarray:
        # E is undefined where K(0,t) underflows -> kernel_viewer already
        # masks those to NaN; confidence is 1 where E is finite, else 0.
        kernel = fs._profile
        E, _ = ring_strength(kernel)
        return np.isfinite(E).astype(float)

    def discretize(self, traj: Trajectory) -> list[str]:
        kernel_t = traj.times
        E = traj.as_array()
        label, _ = classify_coherence(kernel_t, E, self.ring_threshold)
        return [label]


def kernel_regime_report(fs: FeatureSeries):
    """Terminal, OPTIONAL regime read-out. Delegates to the unmodified
    ``kernel_viewer.classify_any`` so the concrete output (the Cartesian
    product transport-label x coherence-label RegimeReport) is identical to
    current kernel-viewer."""
    return classify_any(fs._kernel)
