from dataclasses import dataclass, field
import numpy as np
from .kernel2d import RadialKernel


@dataclass
class Kernel3D:
    """A 3D spatiotemporal kernel K(x, y, z, t).

    Attributes
    ----------
    x, y, z : position arrays for the last three axes of K.
    t : (T,) array of times.
    K : (T, Z, Y, X) array (time axis FIRST, matching the 1D/2D convention).
    origin : (iz, iy, ix) index of the source voxel.

    Memory note: K is held as float64; a 13 x 41^3 kernel is ~90 MB. Store
    float32 on disk and keep L moderate.
    """
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    t: np.ndarray
    K: np.ndarray
    origin: tuple = field(default=None)

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        self.z = np.asarray(self.z, dtype=float)
        self.t = np.asarray(self.t, dtype=float)
        self.K = np.asarray(self.K, dtype=float)
        if self.K.shape != (len(self.t), len(self.z), len(self.y), len(self.x)):
            raise ValueError(f"K shape {self.K.shape} != (T, Z, Y, X)")
        if self.origin is None:
            self.origin = (int(np.argmin(np.abs(self.z))),
                           int(np.argmin(np.abs(self.y))),
                           int(np.argmin(np.abs(self.x))))
        iz, iy, ix = self.origin
        dx = self.x - self.x[ix]; dy = self.y - self.y[iy]; dz = self.z - self.z[iz]
        DZ, DY, DX = np.meshgrid(dz, dy, dx, indexing="ij")
        self._DX, self._DY, self._DZ = DX, DY, DZ
        self._R = np.sqrt(DX ** 2 + DY ** 2 + DZ ** 2)
        self.r_max = float(min(dx.max(), -dx.min(), dy.max(), -dy.min(), dz.max(), -dz.min()))

    # ---------- full-3D quantities ----------
    def k0(self):
        iz, iy, ix = self.origin
        return self.K[:, iz, iy, ix].copy()

    def total(self):
        return self.K.sum(axis=(1, 2, 3))

    def sigma2(self):
        """<x^2 + y^2 + z^2> of |K| on the full 3D kernel."""
        A = np.abs(self.K)
        s = A.sum(axis=(1, 2, 3))
        s = np.where(s == 0, np.nan, s)
        return (A * (self._R ** 2)[None]).sum(axis=(1, 2, 3)) / s

    def edge_fraction(self):
        """Mean |K| on the inscribed-sphere boundary shell over the spatial max
        |K| at each time (robust for front kernels with K(0,t) -> 0)."""
        shell = (self._R >= self.r_max - 1.0) & (self._R <= self.r_max)
        mx = np.abs(self.K).max(axis=(1, 2, 3))
        mx = np.where(mx == 0, np.nan, mx)
        return np.abs(self.K[:, shell]).mean(axis=1) / mx

    def noise_floor(self):
        far = self._R >= 2.0
        if self.t[0] > 0 or not far.any():
            return 1e-3 * float(np.abs(self.K).max())
        return max(float(np.abs(self.K[0][far]).max()), 1e-12)

    # ---------- anisotropy ----------
    def fractional_anisotropy(self):
        """FA(t) of the covariance eigenvalues of |K| (DTI convention):
        0 = isotropic, 1 = a line."""
        A = np.abs(self.K)
        out = np.empty(len(self.t))
        D = np.stack([self._DX, self._DY, self._DZ])
        for i in range(len(self.t)):
            w = A[i]; s = w.sum()
            if s == 0:
                out[i] = np.nan; continue
            M = np.empty((3, 3))
            for a in range(3):
                for b in range(a, 3):
                    M[a, b] = M[b, a] = (w * D[a] * D[b]).sum() / s
            lam = np.linalg.eigvalsh(M)
            lb = lam.mean()
            denom = (lam ** 2).sum()
            out[i] = np.sqrt(1.5 * ((lam - lb) ** 2).sum() / denom) if denom > 0 else np.nan
        return out

    def cubic_anisotropy(self, r_min=2.0):
        """Q4(t) = <n_x^4 + n_y^4 + n_z^4> - 3/5 with weight |K| on voxels with
        r >= r_min. 0 = isotropic; +0.4 = axis-aligned (octahedral front);
        negative = body-diagonal. r_min >= 2 excludes the 6-fold
        nearest-neighbour shell, which fakes Q4 = 0.4 for any narrow kernel."""
        mask = self._R >= r_min
        n4 = (self._DX ** 4 + self._DY ** 4 + self._DZ ** 4)[mask] / np.maximum(self._R[mask] ** 4, 1e-12)
        out = np.empty(len(self.t))
        for i in range(len(self.t)):
            w = np.abs(self.K[i])[mask]; s = w.sum()
            out[i] = (w * n4).sum() / s - 0.6 if s > 0 else np.nan
        return out

    # ---------- radial reduction ----------
    def radial(self):
        """Spherical-shell average up to the inscribed sphere, as a
        RadialKernel with full-3D sigma2/edge/floor and true bin counts."""
        nb = int(np.floor(self.r_max)) + 1
        idx = np.round(self._R).astype(int)
        inside = idx < nb
        flat_idx = idx[inside].ravel()
        counts = np.bincount(flat_idx, minlength=nb)
        P = np.full((len(self.t), nb), np.nan)
        for i in range(len(self.t)):
            sums = np.bincount(flat_idx, weights=self.K[i][inside].ravel(), minlength=nb)
            P[i] = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
        rk = RadialKernel(r=np.arange(nb, dtype=float), t=self.t.copy(), K=P, origin=0)
        rk._sigma2 = self.sigma2()
        rk._edge = self.edge_fraction()
        rk._floor = self.noise_floor()
        rk.bin_counts = counts
        rk.geometry_dim = 3
        return rk
