from dataclasses import dataclass
import numpy as np
from .transforms import cosine_transform_r, cosine_transform_t


@dataclass
class Spectrum:
    k: np.ndarray
    omega: np.ndarray
    S_kw: np.ndarray  # (len(omega), len(k)); S_kw[a, b] = S(k[b], omega[a])

    def peak_frequency(self):
        """omega_peak(k): frequency carrying the most weight at each k."""
        return self.omega[np.argmax(self.S_kw, axis=0)]

    def relaxational_fraction(self, om_cut=None):
        """Fraction of |S| weight below om_cut (default 1.5 * d_omega) per k.
        ~1 = central-peak (relaxational) dominated."""
        if om_cut is None:
            om_cut = 1.5 * (self.omega[1] - self.omega[0])
        A = np.abs(self.S_kw)
        low = A[self.omega <= om_cut].sum(axis=0)
        tot = A.sum(axis=0)
        tot = np.where(tot == 0, np.nan, tot)
        return low / tot


def compute_spectrum(kernel, n_k=25, n_omega=31, omega_max=None, window="hann"):
    """Estimate S(k, omega) from K(r, t) by double cosine transform.

    Resolution: d_omega ~ pi / t_max — short series give coarse omega
    resolution; peak positions are indicative, not precise."""
    rho, P = kernel.symmetrized()
    kvals = np.linspace(0, np.pi, n_k)
    Ck = cosine_transform_r(rho, P, kvals)
    if omega_max is None:
        dt = kernel.t[1] - kernel.t[0] if len(kernel.t) > 1 else 1.0
        omega_max = np.pi / dt
    omegas = np.linspace(0, omega_max, n_omega)
    S = cosine_transform_t(kernel.t, Ck, omegas, window=window)
    return Spectrum(k=kvals, omega=omegas, S_kw=S)


def compute_spectrum_radial(radial_kernel, dim=None, n_k=25, n_omega=31,
                            omega_max=None, window="hann"):
    """S(k, omega) for an ISOTROPIC kernel from its radial profile.

    Spatial transform uses the dimensionally correct radial basis, weighted by
    the TRUE lattice bin counts N(rho) (more faithful than the continuum
    measure 2 pi r dr / 4 pi r^2 dr on small grids):

        C(k, t) = sum_rho N(rho) P(rho, t) B_dim(k rho)
        B_2 = J0 (Bessel),  B_3 = j0 = sin(x)/x (spherical Bessel)

    followed by the windowed cosine transform in time (stationarity assumed).
    Anisotropic kernels: this mixes directions -- check the anisotropy
    diagnostics first.
    """
    from scipy.special import j0 as bessel_j0, j1 as bessel_j1
    rho, P = radial_kernel.symmetrized()
    if dim is None:
        dim = getattr(radial_kernel, "geometry_dim", 2)
    counts = getattr(radial_kernel, "bin_counts", None)
    if counts is None:
        counts = {2: 2 * np.pi * np.maximum(rho, 0.5),
                  3: 4 * np.pi * np.maximum(rho, 0.5) ** 2,
                  4: 2 * np.pi ** 2 * np.maximum(rho, 0.5) ** 3}[dim]

    def _b4(x):
        x = np.asarray(x, dtype=float)
        out = np.ones_like(x)
        nz = np.abs(x) > 1e-12
        out[nz] = 2.0 * bessel_j1(x[nz]) / x[nz]
        return out

    # normalized radial plane-wave average B_d(x) = Gamma(d/2) (2/x)^{d/2-1} J_{d/2-1}(x)
    basis = {2: bessel_j0, 3: (lambda x: np.sinc(np.asarray(x) / np.pi)), 4: _b4}[dim]
    kvals = np.linspace(0, np.pi, n_k)
    Pz = np.nan_to_num(P)
    Ck = np.empty((P.shape[0], n_k))
    for b, k in enumerate(kvals):
        Ck[:, b] = (Pz * (counts * basis(k * rho))[None, :]).sum(axis=1)
    if omega_max is None:
        dt = radial_kernel.t[1] - radial_kernel.t[0] if len(radial_kernel.t) > 1 else 1.0
        omega_max = np.pi / dt
    omegas = np.linspace(0, omega_max, n_omega)
    S = cosine_transform_t(radial_kernel.t, Ck, omegas, window=window)
    return Spectrum(k=kvals, omega=omegas, S_kw=S)
