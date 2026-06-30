"""Synthetic and dynamically-generated 3D kernels (deterministic)."""
import numpy as np
from ..core.kernel3d import Kernel3D


def _grid(L):
    x = np.arange(L, dtype=float) - L // 2
    Z, Y, X = np.meshgrid(x, x, x, indexing="ij")
    return x, X, Y, Z


def diffusive3d(L=33, T=11, dt=0.5, D=0.6, weight=0.25):
    x, X, Y, Z = _grid(L)
    R2 = X ** 2 + Y ** 2 + Z ** 2
    t = np.arange(T) * dt
    K = np.zeros((T, L, L, L))
    K[0, L // 2, L // 2, L // 2] = weight
    for i in range(1, T):
        p = np.exp(-R2 / (4 * D * t[i]))
        K[i] = weight * p / p.sum()
    return Kernel3D(x, x, x, t, K)


def ballistic3d(L=41, T=11, dt=0.5, v=3.0, width=1.0, weight=0.25):
    """An expanding spherical shell: a propagating mode's front in 3D."""
    x, X, Y, Z = _grid(L)
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    t = np.arange(T) * dt
    K = np.zeros((T, L, L, L))
    K[0, L // 2, L // 2, L // 2] = weight
    for i in range(1, T):
        p = np.exp(-(R - v * t[i]) ** 2 / (2 * width ** 2))
        K[i] = weight * p / p.sum()
    return Kernel3D(x, x, x, t, K)


def levy3d(L=49, T=11, dt=0.5, c=0.7, weight=0.25):
    """Isotropic 3D Cauchy-type spreading: P(r) ~ s / (r^2 + s^2)^2."""
    x, X, Y, Z = _grid(L)
    R2 = X ** 2 + Y ** 2 + Z ** 2
    t = np.arange(T) * dt
    K = np.zeros((T, L, L, L))
    K[0, L // 2, L // 2, L // 2] = weight
    for i in range(1, T):
        s = c * t[i]
        p = s / (R2 + s ** 2) ** 2
        K[i] = weight * p / p.sum()
    return Kernel3D(x, x, x, t, K)


def cubic3d(L=41, T=11, dt=0.5, v=2.0, width=1.2, weight=0.25):
    """An octahedral front |x|+|y|+|z| = vt: cubic-lattice light-cone
    anisotropy, invisible to the covariance (FA=0) but caught by Q4."""
    x, X, Y, Z = _grid(L)
    D1 = np.abs(X) + np.abs(Y) + np.abs(Z)
    t = np.arange(T) * dt
    K = np.zeros((T, L, L, L))
    K[0, L // 2, L // 2, L // 2] = weight
    for i in range(1, T):
        p = np.exp(-(D1 - v * t[i]) ** 2 / (2 * width ** 2))
        K[i] = weight * p / p.sum()
    return Kernel3D(x, x, x, t, K)


def randomwalk3d(L=33, T=11, dt=0.5, n_walkers=200000, seed=7, weight=0.25):
    """Monte-Carlo simple cubic random walk ensemble (real sampling noise)."""
    rng = np.random.default_rng(seed)
    x, _, _, _ = _grid(L)
    t = np.arange(T) * dt
    pos = np.zeros((n_walkers, 3), dtype=np.int64)
    K = np.zeros((T, L, L, L))
    c = L // 2

    def deposit(i):
        p = np.clip(pos + c, 0, L - 1)
        H = np.zeros((L, L, L))
        np.add.at(H, (p[:, 2], p[:, 1], p[:, 0]), 1.0)
        K[i] = weight * H / n_walkers

    deposit(0)
    for i in range(1, T):
        for _ in range(2):
            ax = rng.integers(0, 3, size=n_walkers)
            sg = rng.integers(0, 2, size=n_walkers) * 2 - 1
            for a in range(3):
                pos[:, a] += sg * (ax == a)
        deposit(i)
    return Kernel3D(x, x, x, t, K)
