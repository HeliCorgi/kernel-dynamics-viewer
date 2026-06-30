import numpy as np
import matplotlib.pyplot as plt
from ..metrics.ring import ring_strength
from ..metrics.unimodality import violation_amplitude


def plot_ring(kernel, ax=None, threshold=0.05):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.4))
    E, _ = ring_strength(kernel)
    V = violation_amplitude(kernel)
    ef = kernel.edge_fraction()
    clean = np.isfinite(ef) & (ef <= 0.3)
    ax.plot(kernel.t[clean], E[clean], "o-", ms=3, color="#c44", label="ring strength E(t)")
    ax.plot(kernel.t[~clean], E[~clean], "o", ms=3, color="#c44", alpha=0.25)
    ax.plot(kernel.t[clean], V[clean], "s-", ms=3, color="#888", label="violation V(t)")
    ax.axhline(threshold, color="k", ls=":", lw=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("t"); ax.set_ylabel("E, V")
    ax.set_title("ring / monotonicity-violation (faded = boundary-contaminated)")
    ax.legend(fontsize=8)
    return ax
