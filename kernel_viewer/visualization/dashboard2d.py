import numpy as np
import matplotlib.pyplot as plt
from ..core.transforms import local_loglog_slope
from ..metrics.ring import ring_strength
from ..metrics.unimodality import violation_amplitude


def plot_dashboard2d(kernel2d, report=None, path=None):
    """2D summary: late-time snapshot, radial profile heatmap,
    ring/violation/anisotropy curves, transport exponent."""
    rk = kernel2d.radial()
    ef = kernel2d.edge_fraction()
    clean = np.where(np.isfinite(ef) & (ef <= 0.3))[0]
    i_snap = int(clean[-1]) if len(clean) else len(kernel2d.t) - 1

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    a = ax[0, 0]
    ext = [kernel2d.x[0], kernel2d.x[-1], kernel2d.y[0], kernel2d.y[-1]]
    im = a.imshow(kernel2d.K[i_snap], origin="lower", extent=ext, cmap="magma")
    a.set_title(f"K(x, y) at t = {kernel2d.t[i_snap]:g} (latest clean frame)")
    a.set_xlabel("x"); a.set_ylabel("y"); plt.colorbar(im, ax=a)

    a = ax[0, 1]
    rho, P = rk.symmetrized()
    rowmax = np.nanmax(np.abs(P), axis=1, keepdims=True)
    rowmax[rowmax == 0] = 1.0
    im = a.pcolormesh(rho, kernel2d.t, P / rowmax, shading="auto", cmap="magma")
    a.set_xlabel("r (radial bin)"); a.set_ylabel("t")
    a.set_title("radial profile P(r, t) / max_r (row-normalized)")
    plt.colorbar(im, ax=a)

    a = ax[1, 0]
    E, _ = ring_strength(rk)
    V = violation_amplitude(rk)
    A = kernel2d.anisotropy()
    c4 = kernel2d.angular_harmonics()[4]
    m = np.zeros(len(kernel2d.t), dtype=bool); m[clean] = True
    a.plot(kernel2d.t[m], E[m], "o-", ms=3, color="#c44", label="ring E(t)")
    a.plot(kernel2d.t[m], V[m], "s-", ms=3, color="#888", label="violation V(t)")
    a.plot(kernel2d.t[m], A[m], "^-", ms=3, color="#36c", label="anisotropy A(t)")
    a.plot(kernel2d.t[m], c4[m], "v-", ms=3, color="#2a7", label="c4(t), r>=2")
    a.axhline(0, color="k", lw=0.5)
    a.set_yscale("symlog", linthresh=1.0)
    vals = np.concatenate([E[m], V[m], A[m], c4[m]])
    vals = vals[np.isfinite(vals)]
    a.set_xlabel("t")
    a.set_title("shape / anisotropy (clean window; symlog: E,V spike where K(0,t) is tiny)", fontsize=9)
    a.legend(fontsize=7.5)

    a = ax[1, 1]
    tm, sl = local_loglog_slope(kernel2d.t, kernel2d.sigma2())
    a.plot(tm, sl, "o-", ms=4, color="#36c")
    for val, lab in ((1.0, "diffusive"), (2.0, "ballistic")):
        a.axhline(val, color="k", ls=":", lw=0.8)
        a.annotate(lab, (tm[0] if len(tm) else 0, val + 0.03), fontsize=7)
    a.set_xlabel("t"); a.set_ylabel(r"d log $\sigma^2$ / d log t")
    a.set_ylim(0, 2.4)
    title = "transport exponent (full-2D moments)"
    if report is not None:
        title += f"  |  {report.regime.value} / {report.coherence} (p={report.exponent:.2f})"
    a.set_title(title, fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
        return path
    return fig
