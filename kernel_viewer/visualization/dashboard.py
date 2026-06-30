import numpy as np
import matplotlib.pyplot as plt
from ..core.transforms import local_loglog_slope
from .heatmap import plot_heatmap
from .ring_plot import plot_ring
from .spectrum import plot_spectrum


def plot_dashboard(kernel, report=None, path=None):
    """One-figure summary: heatmap, ring/violation, S(k,w), sigma^2 exponent."""
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    plot_heatmap(kernel, ax=ax[0, 0])
    plot_ring(kernel, ax=ax[0, 1])
    plot_spectrum(kernel, ax=ax[1, 0])
    s2 = kernel.sigma2()
    tm, sl = local_loglog_slope(kernel.t, s2)
    a = ax[1, 1]
    a.plot(tm, sl, "o-", ms=4, color="#36c")
    for val, lab in ((1.0, "diffusive"), (2.0, "ballistic")):
        a.axhline(val, color="k", ls=":", lw=0.8)
        a.annotate(lab, (tm[0] if len(tm) else 0, val + 0.03), fontsize=7)
    a.set_xlabel("t"); a.set_ylabel(r"d log $\sigma^2$ / d log t")
    a.set_ylim(0, 2.4)
    title = "local transport exponent"
    if report is not None:
        title += f"  |  {report.regime.value} / {report.coherence} (p={report.exponent:.2f})"
    a.set_title(title, fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return path
    return fig
