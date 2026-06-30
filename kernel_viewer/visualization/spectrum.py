import numpy as np
import matplotlib.pyplot as plt
from ..core.spectral import compute_spectrum


def plot_spectrum(kernel, ax=None, spectrum=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    sp = spectrum or compute_spectrum(kernel)
    A = np.abs(sp.S_kw)
    im = ax.pcolormesh(sp.k / np.pi, sp.omega, A, shading="auto", cmap="viridis")
    ax.plot(sp.k / np.pi, sp.peak_frequency(), "w.--", lw=1, ms=4, label=r"$\omega_{peak}(k)$")
    ax.set_xlabel(r"k / $\pi$"); ax.set_ylabel(r"$\omega$")
    ax.set_title(r"|S(k, $\omega$)|  (central ridge = relaxational)")
    ax.legend(fontsize=8, loc="upper left")
    plt.colorbar(im, ax=ax)
    return ax


def plot_structure_factor(kernel, ax=None, **kw):
    """S(k, omega) colormap + dispersion overlay via the dimensionally correct
    transform (see metrics/fourier.py). Title shows the transform used."""
    from ..metrics.fourier import compute_structure_factor
    sp = compute_structure_factor(kernel, **kw)
    rk = kernel.radial() if hasattr(kernel, "radial") else kernel
    ax = plot_spectrum(rk, ax=ax, spectrum=sp)
    ax.set_title(f"|S(k, $\\omega$)| via {sp.transform_type}"
                 + ("  [ANISOTROPY WARNING]" if sp.warnings else ""), fontsize=9)
    return ax
