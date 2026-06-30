import numpy as np
import matplotlib.pyplot as plt
from ..core.transforms import local_loglog_slope
from ..metrics.ring import ring_strength
from ..metrics.unimodality import violation_amplitude


def plot_dashboard3d(kernel3d, report=None, path=None):
    """3D summary: central z-slice at the latest clean time, row-normalized
    radial profile, shape/anisotropy curves, transport exponent."""
    rk = kernel3d.radial()
    ef = kernel3d.edge_fraction()
    clean = np.where(np.isfinite(ef) & (ef <= 0.3))[0]
    i_snap = int(clean[-1]) if len(clean) else len(kernel3d.t) - 1
    iz = kernel3d.origin[0]

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    a = ax[0, 0]
    ext = [kernel3d.x[0], kernel3d.x[-1], kernel3d.y[0], kernel3d.y[-1]]
    im = a.imshow(kernel3d.K[i_snap, iz], origin="lower", extent=ext, cmap="magma")
    a.set_title(f"K(x, y, z=0) at t = {kernel3d.t[i_snap]:g} (central slice, latest clean frame)")
    a.set_xlabel("x"); a.set_ylabel("y"); plt.colorbar(im, ax=a)

    a = ax[0, 1]
    rho, P = rk.symmetrized()
    rowmax = np.nanmax(np.abs(P), axis=1, keepdims=True); rowmax[rowmax == 0] = 1.0
    im = a.pcolormesh(rho, kernel3d.t, P / rowmax, shading="auto", cmap="magma")
    a.set_xlabel("r (spherical shell)"); a.set_ylabel("t")
    a.set_title("radial profile P(r, t) / max_r (row-normalized)")
    plt.colorbar(im, ax=a)

    a = ax[1, 0]
    E, _ = ring_strength(rk); V = violation_amplitude(rk)
    FA = kernel3d.fractional_anisotropy(); Q4 = kernel3d.cubic_anisotropy()
    m = np.zeros(len(kernel3d.t), dtype=bool); m[clean] = True
    a.plot(kernel3d.t[m], E[m], "o-", ms=3, color="#c44", label="ring E(t)")
    a.plot(kernel3d.t[m], V[m], "s-", ms=3, color="#888", label="violation V(t)")
    a.plot(kernel3d.t[m], FA[m], "^-", ms=3, color="#36c", label="FA(t)")
    a.plot(kernel3d.t[m], Q4[m], "v-", ms=3, color="#2a7", label="Q4(t), r>=2")
    a.axhline(0, color="k", lw=0.5)
    a.set_yscale("symlog", linthresh=1.0)
    vals = np.concatenate([E[m], V[m], FA[m], Q4[m]]); vals = vals[np.isfinite(vals)]
    a.set_xlabel("t"); a.set_title("shape / anisotropy (clean window; symlog: E,V spike where K(0,t) is tiny)", fontsize=9)
    a.legend(fontsize=7.5)

    a = ax[1, 1]
    tm, sl = local_loglog_slope(kernel3d.t, kernel3d.sigma2())
    a.plot(tm, sl, "o-", ms=4, color="#36c")
    for val, lab in ((1.0, "diffusive"), (2.0, "ballistic")):
        a.axhline(val, color="k", ls=":", lw=0.8)
        a.annotate(lab, (tm[0] if len(tm) else 0, val + 0.03), fontsize=7)
    a.set_xlabel("t"); a.set_ylabel(r"d log $\sigma^2$ / d log t"); a.set_ylim(0, 2.4)
    title = "transport exponent (full-3D moments)"
    if report is not None:
        title += f"  |  {report.regime.value} / {report.coherence} (p={report.exponent:.2f})"
    a.set_title(title, fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
        return path
    return fig
