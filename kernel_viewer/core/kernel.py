from dataclasses import dataclass, field
import numpy as np


@dataclass
class Kernel:
    """A spatiotemporal kernel K(r, t).

    Attributes
    ----------
    r : (R,) array of spatial positions (may be signed, e.g. -8..7).
    t : (T,) array of times (non-negative, increasing).
    K : (T, R) array, K[i, j] = K(r=r[j], t=t[i]).
    origin : index into r of the source site (default: argmin |r|).
    """
    r: np.ndarray
    t: np.ndarray
    K: np.ndarray
    origin: int = field(default=-1)

    def __post_init__(self):
        self.r = np.asarray(self.r, dtype=float)
        self.t = np.asarray(self.t, dtype=float)
        self.K = np.asarray(self.K, dtype=float)
        if self.K.shape != (len(self.t), len(self.r)):
            raise ValueError(f"K shape {self.K.shape} != (T={len(self.t)}, R={len(self.r)})")
        if self.origin < 0:
            self.origin = int(np.argmin(np.abs(self.r)))

    def symmetrized(self):
        """Return (rho, P): rho = 0..rmax separations, P[i, m] = mean of K
        over the +-rho pair at each time."""
        c = self.origin
        R = len(self.r)
        rmax = max(c, R - 1 - c)
        P = np.full((len(self.t), rmax + 1), np.nan)
        for m in range(rmax + 1):
            cols = [j for j in (c - m, c + m) if 0 <= j < R]
            P[:, m] = self.K[:, cols].mean(axis=1)
        return np.arange(rmax + 1, dtype=float), P

    def k0(self):
        """K(0, t): the source-site value."""
        return self.K[:, self.origin].copy()

    def total(self):
        """Sum_r K(r,t): constant iff the kernel weight is conserved."""
        return self.K.sum(axis=1)

    def sigma2(self):
        """Second moment of |K| around the origin (transport width)."""
        x = self.r - self.r[self.origin]
        A = np.abs(self.K)
        s = A.sum(axis=1)
        s = np.where(s == 0, np.nan, s)
        return (A * x ** 2).sum(axis=1) / s

    def edge_fraction(self):
        """|K| at the boundary sites over the spatial max |K| at each time;
        >~0.3 signals boundary contamination. (Normalizing by the spatial max
        rather than K(0,t) keeps the measure meaningful for shell/front
        kernels whose source-site value decays to zero.)"""
        mx = np.abs(self.K).max(axis=1)
        mx = np.where(mx == 0, np.nan, mx)
        return 0.5 * (np.abs(self.K[:, 0]) + np.abs(self.K[:, -1])) / mx

    def noise_floor(self):
        """Noise estimate from the t=0 profile away from the origin (where a
        local probe's kernel should vanish); fallback: 1e-3 * max|K|."""
        c = self.origin
        away = [j for j in range(len(self.r)) if abs(j - c) >= 2]
        if not away or self.t[0] > 0:
            return 1e-3 * float(np.abs(self.K).max())
        v = float(np.abs(self.K[0, away]).max())
        return max(v, 1e-12)
