"""Unified dynamic-structure-factor API (v0.4).

Dispatches to the dimensionally correct spatial transform:
  1D : symmetric cosine transform               (transform_type "cosine")
  2D : Hankel, order 0  -- J0(kr)               ("hankel_j0")
  3D : spherical Bessel, n=0 -- j0(kr)=sin/x    ("spherical_bessel_j0")
  4D : generalized Hankel -- 2 J1(kr)/(kr)      ("generalized_hankel_j1")
followed by the Hann-windowed cosine transform in time (stationary kernels:
C(-t) = C(t)).

Discretization note: instead of the continuum quadrature
2*pi Int r dr J0(kr) P(r) (resp. 4*pi Int r^2 dr j0 P), the 2D/3D transforms
weight each radial bin by its TRUE lattice pixel/voxel count N(rho). On small
integer grids this is more faithful than the continuum measure, handles r=0
without any singularity (N(0)=1, basis(0)=1), and gives the exact sum rule
C(k=0, t) = sum_pixels K. Validated against the exact Gaussian symbol
C(k,t) = W exp(-D k^2 t) to ~10% (integer-bin discretization).
"""
import numpy as np
from ..core.kernel import Kernel
from ..core.kernel2d import Kernel2D, RadialKernel
from ..core.kernel3d import Kernel3D
from ..core.kernel4d import Kernel4D
from ..core.spectral import compute_spectrum, compute_spectrum_radial


def compute_structure_factor(kernel, dim=None, use_hankel=True, n_k=25,
                             n_omega=31, omega_max=None, window="hann",
                             isotropy_threshold=0.9):
    """Compute S(k, omega) for a Kernel / Kernel2D / Kernel3D (or an already
    radially reduced RadialKernel).

    Returns a Spectrum with two extra attributes:
      .transform_type : "cosine" / "hankel_j0" / "spherical_bessel_j0"
      .warnings       : list of strings (e.g. anisotropy warning)

    For 2D/3D input the kernel is radially reduced internally; if the kernel's
    isotropy_score (1 - max anisotropy) falls below `isotropy_threshold`, the
    radial transform still runs (there is no honest anisotropic alternative in
    this package) but a warning is attached -- the radial S(k, omega) then
    mixes directions. Set use_hankel=False to force the 1D cosine transform on
    the radial profile (NOT recommended; wrong measure in 2D/3D; provided for
    comparison only).
    """
    warnings = []
    if isinstance(kernel, (Kernel2D, Kernel3D, Kernel4D)):
        d = 2 if isinstance(kernel, Kernel2D) else (3 if isinstance(kernel, Kernel3D) else 4)
        # isotropy check before reduction
        if isinstance(kernel, (Kernel3D, Kernel4D)) and not isinstance(kernel, Kernel2D):
            fa = float(np.nanmax(kernel.fractional_anisotropy()[1:])) if len(kernel.t) > 1 else 0.0
            qiso = 0.4 if d == 3 else 0.5
            q4 = float(np.nanmedian(kernel.cubic_anisotropy()[len(kernel.t) // 2:]))
            score = float(np.clip(1.0 - max(fa, abs(q4) / qiso), 0, 1))
        elif isinstance(kernel, Kernel2D):
            a = float(np.nanmax(kernel.anisotropy()[1:])) if len(kernel.t) > 1 else 0.0
            c4 = float(np.nanmedian(kernel.angular_harmonics()[4][len(kernel.t) // 2:]))
            score = float(np.clip(1.0 - max(a, c4), 0, 1))
        if score < isotropy_threshold:
            warnings.append(f"isotropy_score={score:.2f} < {isotropy_threshold}: "
                            "radial S(k,omega) mixes directions")
        rk = kernel.radial()
    elif isinstance(kernel, RadialKernel):
        rk = kernel
        d = getattr(rk, "geometry_dim", 2)
    else:
        sp = compute_spectrum(kernel, n_k=n_k, n_omega=n_omega,
                              omega_max=omega_max, window=window)
        sp.transform_type = "cosine"
        sp.warnings = warnings
        return sp
    if dim is not None:
        d = dim
    if not use_hankel:
        warnings.append("use_hankel=False: 1D cosine transform on a radial "
                        "profile has the wrong measure in 2D/3D")
        sp = compute_spectrum(rk, n_k=n_k, n_omega=n_omega,
                              omega_max=omega_max, window=window)
        sp.transform_type = "cosine"
    else:
        sp = compute_spectrum_radial(rk, dim=d, n_k=n_k, n_omega=n_omega,
                                     omega_max=omega_max, window=window)
        sp.transform_type = {2: "hankel_j0", 3: "spherical_bessel_j0",
                             4: "generalized_hankel_j1"}[d]
    sp.warnings = warnings
    return sp
