from dataclasses import dataclass, field
import numpy as np
from .kernel import Kernel


@dataclass
class Kernel2D:
    """A 2D spatiotemporal kernel K(x, y, t).

    Attributes
    ----------
    x : (X,) array of positions along the second image axis.
    y : (Y,) array of positions along the first image axis.
    t : (T,) array of times.
    K : (T, Y, X) array, K[i, jy, jx] = K(x=x[jx], y=y[jy], t=t[i]).
    origin : (iy, ix) index of the source pixel (default: argmin |x|, |y|).
    """
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    K: np.ndarray
    origin: tuple = field(default=None)

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        self.t = np.asarray(self.t, dtype=float)
        self.K = np.asarray(self.K, dtype=float)
        if self.K.shape != (len(self.t), len(self.y), len(self.x)):
            raise ValueError(f"K shape {self.K.shape} != (T, Y, X) = "
                             f"({len(self.t)}, {len(self.y)}, {len(self.x)})")
        if self.origin is None:
            self.origin = (int(np.argmin(np.abs(self.y))), int(np.argmin(np.abs(self.x))))
        iy, ix = self.origin
        dx = self.x - self.x[ix]
        dy = self.y - self.y[iy]
        self._DX, self._DY = np.meshgrid(dx, dy)
        self._R = np.sqrt(self._DX ** 2 + self._DY ** 2)
        # inscribed-circle radius: corners of a square window are sampled
        # anisotropically, so radial statistics stop at the nearest edge
        self.r_max = float(min(dx.max(), -dx.min(), dy.max(), -dy.min()))

    # ---------- full-2D quantities (correct measure) ----------
    def k0(self):
        iy, ix = self.origin
        return self.K[:, iy, ix].copy()

    def total(self):
        return self.K.sum(axis=(1, 2))

    def sigma2(self):
        """<x^2 + y^2> of |K| (computed on the full 2D kernel, NOT from the
        radial profile, which would miss the r dr measure)."""
        A = np.abs(self.K)
        s = A.sum(axis=(1, 2))
        s = np.where(s == 0, np.nan, s)
        return (A * (self._R ** 2)[None]).sum(axis=(1, 2)) / s

    def edge_fraction(self):
        """Mean |K| on the inscribed-circle boundary ring over the spatial max
        |K| at each time (robust for front kernels with K(0,t) -> 0)."""
        ring = (self._R >= self.r_max - 1.0) & (self._R <= self.r_max)
        mx = np.abs(self.K).max(axis=(1, 2))
        mx = np.where(mx == 0, np.nan, mx)
        return np.abs(self.K[:, ring]).mean(axis=1) / mx

    def noise_floor(self):
        far = self._R >= 2.0
        if self.t[0] > 0 or not far.any():
            return 1e-3 * float(np.abs(self.K).max())
        return max(float(np.abs(self.K[0][far]).max()), 1e-12)

    # ---------- anisotropy ----------
    def anisotropy(self):
        """A(t) = (l1 - l2)/(l1 + l2) of the covariance eigenvalues of |K|.
        0 = isotropic spreading; 1 = a line."""
        A = np.abs(self.K)
        out = np.empty(len(self.t))
        for i in range(len(self.t)):
            w = A[i]; s = w.sum()
            if s == 0:
                out[i] = np.nan; continue
            mxx = (w * self._DX ** 2).sum() / s
            myy = (w * self._DY ** 2).sum() / s
            mxy = (w * self._DX * self._DY).sum() / s
            tr = mxx + myy
            det = mxx * myy - mxy ** 2
            disc = max(tr ** 2 / 4 - det, 0.0)
            l1 = tr / 2 + np.sqrt(disc); l2 = tr / 2 - np.sqrt(disc)
            out[i] = (l1 - l2) / tr if tr > 0 else np.nan
        return out

    def angular_harmonics(self, m=(2, 4), r_min=2.0):
        """c_m(t) = |sum w e^{i m theta}| / sum w with w = |K| on pixels with
        r >= r_min. c4 picks up square-lattice (diamond light-cone)
        anisotropy. r_min >= 2 excludes the nearest-neighbour shell, whose
        4-fold geometry fakes c4 ~ 1 for any narrow kernel."""
        theta = np.arctan2(self._DY, self._DX)
        mask = self._R >= r_min
        out = {mm: np.empty(len(self.t)) for mm in m}
        for i in range(len(self.t)):
            w = np.abs(self.K[i])[mask]; s = w.sum()
            for mm in m:
                out[mm][i] = np.abs((w * np.exp(1j * mm * theta[mask])).sum()) / s if s > 0 else np.nan
        return out

    # ---------- radial reduction ----------
    def radial(self):
        """Radially averaged kernel as a RadialKernel (drop-in for the 1D
        machinery). Profile bins are unit-width annuli up to the inscribed
        circle; sigma2/edge_fraction/noise_floor are overridden with the
        full-2D values so transport is measured with the correct measure."""
        nb = int(np.floor(self.r_max)) + 1
        idx = np.round(self._R).astype(int)
        inside = idx < nb
        P = np.full((len(self.t), nb), np.nan)
        counts = np.bincount(idx[inside].ravel(), minlength=nb)
        for i in range(len(self.t)):
            sums = np.bincount(idx[inside].ravel(), weights=self.K[i][inside].ravel(), minlength=nb)
            with np.errstate(invalid="ignore"):
                P[i] = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
        rk = RadialKernel(r=np.arange(nb, dtype=float), t=self.t.copy(), K=P, origin=0)
        rk._sigma2 = self.sigma2()
        rk._edge = self.edge_fraction()
        rk._floor = self.noise_floor()
        rk.bin_counts = counts
        rk.geometry_dim = 2
        return rk


class RadialKernel(Kernel):
    """A radial profile P(r, t) carrying full-2D moments. r is the radial
    coordinate (origin index 0); symmetrization is the identity."""
    _sigma2 = None
    _edge = None
    _floor = None

    def symmetrized(self):
        return self.r.copy(), self.K.copy()

    def sigma2(self):
        return self._sigma2 if self._sigma2 is not None else super().sigma2()

    def edge_fraction(self):
        return self._edge if self._edge is not None else super().edge_fraction()

    def noise_floor(self):
        return self._floor if self._floor is not None else super().noise_floor()
