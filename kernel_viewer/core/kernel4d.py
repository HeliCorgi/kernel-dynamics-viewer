from dataclasses import dataclass, field
import numpy as np
from .kernel2d import RadialKernel


@dataclass
class Kernel4D:
    """A 4D spatiotemporal kernel K(x, y, z, w, t).

    Attributes
    ----------
    x, y, z, w : position arrays for the last four axes of K.
    t : (T,) array of times.
    K : (T, W, Z, Y, X) array (time axis FIRST, as in 1D/2D/3D).
    origin : (iw, iz, iy, ix) index of the source voxel.

    Memory note: 4D kernels are large -- a 11 x 25^4 kernel is ~34 MB as
    float64 and the distance grids add several more copies of 25^4. Store
    float32 on disk, keep L <= ~25, and expect the inscribed-hypersphere
    radial range to be short (r_max = L//2).
    """
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    w: np.ndarray
    t: np.ndarray
    K: np.ndarray
    origin: tuple = field(default=None)

    def __post_init__(self):
        for name in ("x", "y", "z", "w", "t"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        self.K = np.asarray(self.K, dtype=float)
        if self.K.shape != (len(self.t), len(self.w), len(self.z), len(self.y), len(self.x)):
            raise ValueError(f"K shape {self.K.shape} != (T, W, Z, Y, X)")
        if self.origin is None:
            self.origin = (int(np.argmin(np.abs(self.w))), int(np.argmin(np.abs(self.z))),
                           int(np.argmin(np.abs(self.y))), int(np.argmin(np.abs(self.x))))
        iw, iz, iy, ix = self.origin
        dx = self.x - self.x[ix]; dy = self.y - self.y[iy]
        dz = self.z - self.z[iz]; dw = self.w - self.w[iw]
        DW, DZ, DY, DX = np.meshgrid(dw, dz, dy, dx, indexing="ij")
        self._D = (DX, DY, DZ, DW)
        self._R = np.sqrt(DX ** 2 + DY ** 2 + DZ ** 2 + DW ** 2)
        self.r_max = float(min(dx.max(), -dx.min(), dy.max(), -dy.min(),
                               dz.max(), -dz.min(), dw.max(), -dw.min()))

    # ---------- full-4D quantities (exact voxel distances) ----------
    def k0(self):
        iw, iz, iy, ix = self.origin
        return self.K[:, iw, iz, iy, ix].copy()

    def total(self):
        return self.K.sum(axis=(1, 2, 3, 4))

    def sigma2(self):
        """<x^2+y^2+z^2+w^2> of |K| on the full 4D kernel (= 8 D t for
        Gaussian diffusion: sigma2 = 2 d D t with d = 4)."""
        A = np.abs(self.K)
        s = A.sum(axis=(1, 2, 3, 4))
        s = np.where(s == 0, np.nan, s)
        return (A * (self._R ** 2)[None]).sum(axis=(1, 2, 3, 4)) / s

    def edge_fraction(self):
        """Mean |K| on the inscribed-hypersphere boundary shell over the
        spatial max |K| at each time (robust for front kernels)."""
        shell = (self._R >= self.r_max - 1.0) & (self._R <= self.r_max)
        mx = np.abs(self.K).max(axis=(1, 2, 3, 4))
        mx = np.where(mx == 0, np.nan, mx)
        return np.abs(self.K[:, shell]).mean(axis=1) / mx

    def noise_floor(self):
        far = self._R >= 2.0
        if self.t[0] > 0 or not far.any():
            return 1e-3 * float(np.abs(self.K).max())
        return max(float(np.abs(self.K[0][far]).max()), 1e-12)

    # ---------- anisotropy (hyperoctahedral B4 symmetry) ----------
    def fractional_anisotropy(self):
        """Generalized FA(t) in d=4: sqrt(d/(d-1) * sum(l_i - lbar)^2 /
        sum l_i^2), eigenvalues of the 4x4 covariance of |K|.
        0 = isotropic, 1 = a line (single nonzero eigenvalue)."""
        A = np.abs(self.K)
        out = np.empty(len(self.t))
        for i in range(len(self.t)):
            v = A[i]; s = v.sum()
            if s == 0:
                out[i] = np.nan; continue
            M = np.empty((4, 4))
            for a in range(4):
                for b in range(a, 4):
                    M[a, b] = M[b, a] = (v * self._D[a] * self._D[b]).sum() / s
            lam = np.linalg.eigvalsh(M)
            lb = lam.mean()
            den = (lam ** 2).sum()
            out[i] = np.sqrt((4.0 / 3.0) * ((lam - lb) ** 2).sum() / den) if den > 0 else np.nan
        return out

    def cubic_anisotropy(self, r_min=2.0):
        """Q4(t) = <n_x^4+n_y^4+n_z^4+n_w^4> - 1/2 with weight |K| on voxels
        with r >= r_min. The isotropic value of <sum n_i^4> on S^3 is
        3/(d+2) * d / d = ... = sum_i 3/(d(d+2)) * d = 3/(d+2) = 1/2 for d=4.
        0 = isotropic; +1/2 = axis-aligned (hyperoctahedral front vertices);
        negative = body-diagonal. r_min >= 2 excludes the 8-fold
        nearest-neighbour shell (fakes Q4 = +1/2 for narrow kernels)."""
        mask = self._R >= r_min
        DX, DY, DZ, DW = self._D
        n4 = (DX ** 4 + DY ** 4 + DZ ** 4 + DW ** 4)[mask] / np.maximum(self._R[mask] ** 4, 1e-12)
        out = np.empty(len(self.t))
        for i in range(len(self.t)):
            v = np.abs(self.K[i])[mask]; s = v.sum()
            out[i] = (v * n4).sum() / s - 0.5 if s > 0 else np.nan
        return out

    # ---------- radial reduction (hyperspherical shells) ----------
    def radial(self):
        """Hyperspherical-shell average up to the inscribed hypersphere, as a
        RadialKernel with full-4D sigma2/edge/floor and true bin counts
        (occupancy ~ 2 pi^2 r^3 dr for large r)."""
        nb = int(np.floor(self.r_max)) + 1
        idx = np.round(self._R).astype(int)
        inside = idx < nb
        flat = idx[inside].ravel()
        counts = np.bincount(flat, minlength=nb)
        P = np.full((len(self.t), nb), np.nan)
        for i in range(len(self.t)):
            sums = np.bincount(flat, weights=self.K[i][inside].ravel(), minlength=nb)
            P[i] = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
        rk = RadialKernel(r=np.arange(nb, dtype=float), t=self.t.copy(), K=P, origin=0)
        rk._sigma2 = self.sigma2()
        rk._edge = self.edge_fraction()
        rk._floor = self.noise_floor()
        rk.bin_counts = counts
        rk.geometry_dim = 4
        return rk
