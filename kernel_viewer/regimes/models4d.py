"""Synthetic and dynamically-generated 4D kernels (deterministic).
4D arrays are big: defaults stay at L <= 25; everything float64 in RAM."""
import numpy as np
from ..core.kernel4d import Kernel4D


def _grid(L):
    x = np.arange(L, dtype=float) - L // 2
    W, Z, Y, X = np.meshgrid(x, x, x, x, indexing="ij")
    return x, X, Y, Z, W


def diffusive4d(L=21, T=9, dt=0.5, D=0.5, weight=0.25):
    x, X, Y, Z, W = _grid(L)
    R2 = X ** 2 + Y ** 2 + Z ** 2 + W ** 2
    t = np.arange(T) * dt
    K = np.zeros((T, L, L, L, L))
    K[0, L // 2, L // 2, L // 2, L // 2] = weight
    for i in range(1, T):
        p = np.exp(-R2 / (4 * D * t[i]))
        K[i] = weight * p / p.sum()
    return Kernel4D(x, x, x, x, t, K)


def ballistic4d(L=25, T=9, dt=0.5, v=3.0, width=1.0, weight=0.25):
    """An expanding hyperspherical shell (surface dilution ~ 1/r^3)."""
    x, X, Y, Z, W = _grid(L)
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2 + W ** 2)
    t = np.arange(T) * dt
    K = np.zeros((T, L, L, L, L))
    K[0, L // 2, L // 2, L // 2, L // 2] = weight
    for i in range(1, T):
        p = np.exp(-(R - v * t[i]) ** 2 / (2 * width ** 2))
        K[i] = weight * p / p.sum()
    return Kernel4D(x, x, x, x, t, K)


def levy4d(L=25, T=9, dt=0.5, c=0.6, weight=0.25):
    """Isotropic Cauchy-type spreading in d=4: P(r) ~ s/(r^2+s^2)^{5/2}
    (power-law tail r^{-5})."""
    x, X, Y, Z, W = _grid(L)
    R2 = X ** 2 + Y ** 2 + Z ** 2 + W ** 2
    t = np.arange(T) * dt
    K = np.zeros((T, L, L, L, L))
    K[0, L // 2, L // 2, L // 2, L // 2] = weight
    for i in range(1, T):
        s = c * t[i]
        p = s / (R2 + s ** 2) ** 2.5
        K[i] = weight * p / p.sum()
    return Kernel4D(x, x, x, x, t, K)


def cubic4d(L=25, T=9, dt=0.5, v=1.8, width=1.0, weight=0.25):
    """A hyperoctahedral front |x|+|y|+|z|+|w| = vt: B4 lattice light-cone
    anisotropy -- FA = 0 by symmetry, caught by Q4."""
    x, X, Y, Z, W = _grid(L)
    D1 = np.abs(X) + np.abs(Y) + np.abs(Z) + np.abs(W)
    t = np.arange(T) * dt
    K = np.zeros((T, L, L, L, L))
    K[0, L // 2, L // 2, L // 2, L // 2] = weight
    for i in range(1, T):
        p = np.exp(-(D1 - v * t[i]) ** 2 / (2 * width ** 2))
        K[i] = weight * p / p.sum()
    return Kernel4D(x, x, x, x, t, K)


def randomwalk4d(L=21, T=9, dt=0.5, n_walkers=100000, seed=7, weight=0.25):
    """Monte-Carlo hypercubic random walk ensemble."""
    rng = np.random.default_rng(seed)
    x, _, _, _, _ = _grid(L)
    t = np.arange(T) * dt
    pos = np.zeros((n_walkers, 4), dtype=np.int64)
    K = np.zeros((T, L, L, L, L))
    c = L // 2

    def deposit(i):
        p = np.clip(pos + c, 0, L - 1)
        H = np.zeros((L, L, L, L))
        np.add.at(H, (p[:, 3], p[:, 2], p[:, 1], p[:, 0]), 1.0)
        K[i] = weight * H / n_walkers

    deposit(0)
    for i in range(1, T):
        for _ in range(2):
            ax = rng.integers(0, 4, size=n_walkers)
            sg = rng.integers(0, 2, size=n_walkers) * 2 - 1
            for a in range(4):
                pos[:, a] += sg * (ax == a)
        deposit(i)
    return Kernel4D(x, x, x, x, t, K)
