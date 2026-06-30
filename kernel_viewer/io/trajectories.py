"""Convert trajectory ensembles (e.g. from yupi) into kernels K(r,t).

An ensemble of walkers all starting at the origin IS the propagator of the
process, so histogramming their positions at each time yields K(x,y,t) (a
Kernel2D) for light-tailed processes, or a radial density profile P(r,t) (a 1D
Kernel) for heavy-tailed (Lévy) processes whose rare long jumps a finite square
lattice would clip.

These helpers take plain NumPy arrays and construct the EXISTING Kernel /
Kernel2D objects -- they add no new diagnostics and import no yupi (yupi lives
only in the example generator script).
"""
from __future__ import annotations
import numpy as np
from ..core.kernel import Kernel
from ..core.kernel2d import Kernel2D


def ensemble_to_kernel2d(positions: np.ndarray, L: int = 41,
                         weight: float = 0.25) -> Kernel2D:
    """positions: (N_walkers, T_steps, 2), walkers starting near the origin.

    Square-lattice histogram per time -> Kernel2D. Appropriate for
    light-tailed processes (Brownian / normal diffusion). NOT appropriate for
    Lévy flights: a finite box clips the heavy tail. Use
    ``ensemble_to_radial_kernel`` for those.
    """
    positions = np.asarray(positions, dtype=float)
    N, T, d = positions.shape
    if d != 2:
        raise ValueError("ensemble_to_kernel2d expects (N, T, 2)")
    c = L // 2
    x = np.arange(L) - c
    edges = np.arange(-c - 0.5, c + 1.5)
    K = np.zeros((T, L, L))
    for it in range(T):
        xy = positions[:, it, :]
        H, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], bins=[edges, edges])
        s = H.sum()
        K[it] = weight * H / s if s > 0 else H
    return Kernel2D(x=x, y=x, t=np.arange(T, dtype=float), K=K)


def ensemble_to_radial_kernel(positions: np.ndarray, pct: float = 99.5,
                              weight: float = 0.25) -> Kernel:
    """positions: (N_walkers, T_steps, D), walkers starting near the origin.

    Build a radial DENSITY profile P(r,t) for an (assumed isotropic) process:
    histogram the radial distance out to the ``pct`` percentile, then divide by
    the shell area (2*pi*r, i.e. 2D) to get a density, returned as a 1D
    ``Kernel``. This keeps more of the tail than a fixed square lattice.

    Caveat for very heavy tails: preserving the tail means ``Rmax`` (the ``pct``
    percentile) can grow large, producing a many-bin profile that makes
    kernel-viewer's tail-shape classification slow (its peak-count is ~O(R^2)).
    That heuristic is also not a robust heavy-tail / Lévy detector -- for
    anomalous-diffusion analysis prefer dedicated tools (MSD-exponent fits, van
    Hove functions, ``scipy.stats.levy_stable``) over this profile.
    """
    positions = np.asarray(positions, dtype=float)
    N, T, D = positions.shape
    rad = np.sqrt((positions ** 2).sum(axis=2))            # (N, T)
    Rmax = int(np.percentile(rad[:, -1], pct))
    if Rmax < 4:
        raise ValueError("radial range too small; increase ensemble spread or pct")
    r = np.arange(Rmax + 1, dtype=float)
    shell = np.maximum(2 * np.pi * r, 1.0); shell[0] = 1.0  # 2D shell area
    P = np.zeros((T, Rmax + 1))
    for it in range(T):
        H, _ = np.histogram(rad[:, it], bins=np.arange(-0.5, Rmax + 1.5))
        dens = H / shell
        s = dens.sum()
        P[it] = weight * dens / s if s > 0 else dens
    return Kernel(r=r, t=np.arange(T, dtype=float), K=P)
