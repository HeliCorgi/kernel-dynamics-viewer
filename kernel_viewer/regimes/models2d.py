"""Synthetic and dynamically-generated 2D kernels (deterministic)."""
import numpy as np
from ..core.kernel2d import Kernel2D


def _grid(L):
    x = np.arange(L, dtype=float) - L // 2
    X, Y = np.meshgrid(x, x)
    return x, X, Y


def diffusive2d(L=41, T=13, dt=0.5, D=0.6, weight=0.25):
    x, X, Y = _grid(L)
    t = np.arange(T) * dt
    K = np.zeros((T, L, L))
    K[0, L // 2, L // 2] = weight
    for i in range(1, T):
        p = np.exp(-(X ** 2 + Y ** 2) / (4 * D * t[i]))
        K[i] = weight * p / p.sum()
    return Kernel2D(x, x, t, K)


def ballistic2d(L=61, T=13, dt=0.5, v=2.5, width=1.2, weight=0.25):
    """An expanding circular wave front: in 2D a propagating mode IS a ring."""
    x, X, Y = _grid(L)
    R = np.sqrt(X ** 2 + Y ** 2)
    t = np.arange(T) * dt
    K = np.zeros((T, L, L))
    K[0, L // 2, L // 2] = weight
    for i in range(1, T):
        p = np.exp(-(R - v * t[i]) ** 2 / (2 * width ** 2))
        K[i] = weight * p / p.sum()
    return Kernel2D(x, x, t, K)


def levy2d(L=81, T=13, dt=0.5, c=0.8, weight=0.25):
    """Isotropic 2D Cauchy spreading: P(r) ~ s / (r^2 + s^2)^{3/2}."""
    x, X, Y = _grid(L)
    R2 = X ** 2 + Y ** 2
    t = np.arange(T) * dt
    K = np.zeros((T, L, L))
    K[0, L // 2, L // 2] = weight
    for i in range(1, T):
        s = c * t[i]
        p = s / (R2 + s ** 2) ** 1.5
        K[i] = weight * p / p.sum()
    return Kernel2D(x, x, t, K)


def anisotropic2d(L=61, T=13, dt=0.5, v=2.2, width=1.2, weight=0.25):
    """A diamond (|x|+|y| = vt) front: square-lattice light-cone anisotropy.
    Radial averaging smears this into a broad shoulder -> c4 flags it."""
    x, X, Y = _grid(L)
    D1 = np.abs(X) + np.abs(Y)
    t = np.arange(T) * dt
    K = np.zeros((T, L, L))
    K[0, L // 2, L // 2] = weight
    for i in range(1, T):
        p = np.exp(-(D1 - v * t[i]) ** 2 / (2 * width ** 2))
        K[i] = weight * p / p.sum()
    return Kernel2D(x, x, t, K)


def randomwalk2d(L=41, T=13, dt=0.5, hop=0.5, n_walkers=200000, seed=7, weight=0.25):
    """Monte-Carlo lattice random walk ensemble: genuinely simulated diffusive
    dynamics with realistic sampling noise (not a closed form)."""
    rng = np.random.default_rng(seed)
    x, _, _ = _grid(L)
    t = np.arange(T) * dt
    steps_per_frame = max(int(round(hop * 4 * dt / 1.0 * 4)), 1)
    pos = np.zeros((n_walkers, 2), dtype=np.int64)
    K = np.zeros((T, L, L))
    c = L // 2

    def deposit(i):
        px = np.clip(pos[:, 0] + c, 0, L - 1)
        py = np.clip(pos[:, 1] + c, 0, L - 1)
        H = np.zeros((L, L))
        np.add.at(H, (py, px), 1.0)
        K[i] = weight * H / n_walkers

    deposit(0)
    for i in range(1, T):
        for _ in range(steps_per_frame):
            move = rng.integers(0, 4, size=n_walkers)
            pos[:, 0] += (move == 0).astype(np.int64) - (move == 1)
            pos[:, 1] += (move == 2).astype(np.int64) - (move == 3)
        deposit(i)
    return Kernel2D(x, x, t, K)


def wave2d(L=81, T=13, dt=0.5, c2=2.0, damp=0.0, weight=0.25):
    """Leapfrog integration of the discrete 2D wave equation on a square
    lattice: a genuine lattice dynamics example. Shows a circular front plus
    C4 lattice anisotropy at short wavelengths."""
    x, _, _ = _grid(L)
    t = np.arange(T) * dt
    u = np.zeros((L, L)); u[L // 2, L // 2] = 1.0
    up = u.copy()
    K = np.zeros((T, L, L))
    K[0] = weight * np.abs(u) / np.abs(u).sum()
    sub = 8
    h = dt / sub
    for i in range(1, T):
        for _ in range(sub):
            lap = (np.roll(u, 1, 0) + np.roll(u, -1, 0) + np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4 * u)
            unew = 2 * u - up + c2 * h ** 2 * lap - damp * h * (u - up)
            up, u = u, unew
        A = np.abs(u)
        K[i] = weight * A / A.sum()
    return Kernel2D(x, x, t, K)


def coherent_onset2d(L=33, T=61, dt=0.25, D=0.6, t_on=7.5, om=2.0, r0=3.0,
                     amp=0.8, weight=0.25):
    """Diffusive background for all t; from t_on a recurrent breathing ring
    (a long-lived coherent oscillation) switches on. For testing transition
    detection with track_regimes."""
    x, X, Y = _grid(L)
    R = np.sqrt(X ** 2 + Y ** 2)
    t = np.arange(T) * dt
    K = np.zeros((T, L, L))
    K[0, L // 2, L // 2] = weight
    for i in range(1, T):
        base = np.exp(-R ** 2 / (4 * D * t[i]))
        p = base.copy()
        if t[i] > t_on:
            p = p + amp * (1 - np.cos(om * (t[i] - t_on))) * np.exp(-(R - r0) ** 2)
        K[i] = weight * p / p.sum()
    return Kernel2D(x, x, t, K)
