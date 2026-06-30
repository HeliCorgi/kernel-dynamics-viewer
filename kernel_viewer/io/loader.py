import os
import warnings
import numpy as np
from ..core.kernel import Kernel
from ..core.kernel2d import Kernel2D
from ..core.kernel3d import Kernel3D
from ..core.kernel4d import Kernel4D


def _build(K, d=None):
    K = np.asarray(K, dtype=float)
    get = (lambda k: d[k] if (d is not None and k in d) else None)
    t = get("t")
    if t is None:
        t = np.arange(K.shape[0], dtype=float)
    if K.ndim == 2:
        r = get("r")
        if r is None:
            r = np.arange(K.shape[1]) - K.shape[1] // 2
        return Kernel(r=r, t=t, K=K)
    if K.ndim == 3:
        x = get("x"); y = get("y")
        if x is None:
            x = np.arange(K.shape[2]) - K.shape[2] // 2
        if y is None:
            y = np.arange(K.shape[1]) - K.shape[1] // 2
        return Kernel2D(x=x, y=y, t=t, K=K)
    if K.ndim == 4:
        x = get("x"); y = get("y"); z = get("z")
        if x is None: x = np.arange(K.shape[3]) - K.shape[3] // 2
        if y is None: y = np.arange(K.shape[2]) - K.shape[2] // 2
        if z is None: z = np.arange(K.shape[1]) - K.shape[1] // 2
        return Kernel3D(x=x, y=y, z=z, t=t, K=K)
    if K.ndim == 5:
        ax = {}
        for nm, j in (("x", 4), ("y", 3), ("z", 2), ("w", 1)):
            v = get(nm)
            ax[nm] = v if v is not None else np.arange(K.shape[j]) - K.shape[j] // 2
        return Kernel4D(x=ax["x"], y=ax["y"], z=ax["z"], w=ax["w"], t=t, K=K)
    raise ValueError(f"K must be (T, R), (T, Y, X), (T, Z, Y, X) or (T, W, Z, Y, X); got ndim={K.ndim}")


def load_kernel(path):
    """Load a kernel. .npz keys: K (required; (T,R) for 1D, (T,Y,X) for 2D)
    plus optional t and r (1D) or x, y (2D). Bare .npy: K only, integer axes
    assumed. Returns Kernel or Kernel2D by dimensionality."""
    if str(path).endswith(".npz"):
        d = np.load(path)
        K = d["K"] if "K" in d else d[list(d.keys())[0]]
        return _build(K, d)
    return _build(np.load(path))


# ---------------------------------------------------------------------------
# v0.8: load_any -- a superset loader that also reads .mat and .csv/.txt.
# load_kernel and _build above are left byte-for-byte unchanged for backward
# compatibility; load_any delegates to them for .npz/.npy and reuses _build()
# for every format, so the resulting Kernel objects are identical regardless
# of input file type. No new analysis math lives here -- only parsing.
# ---------------------------------------------------------------------------

_SPATIAL_KEYS = ("r", "x", "y", "z", "w")


def load_any(path, time_axis=None, layout="auto", **axes_hints):
    """Load a kernel from .npz/.npy/.mat/.csv/.txt.

    A strict superset of :func:`load_kernel`: ``.npz``/``.npy`` are delegated
    unchanged. New formats:

    * ``.mat`` (``scipy.io.loadmat``): the kernel array is the variable named
      ``K`` if present, else the highest-dimensional non-private array. Optional
      coordinate vectors ``t`` and ``r``/``x``/``y``/``z``/``w`` are read if
      present. MATLAB is column-major and often stores time last (``(R, T)`` or
      ``(X, Y, T)``); the time axis is located from the coordinate lengths and
      moved to axis 0. Pass ``time_axis='first'`` or ``'last'`` to override an
      ambiguous case.
    * ``.csv``/``.txt`` (``numpy.genfromtxt``): **1D kernels only**, ``K(r, t)``.
      ``layout='matrix'|'long'|'auto'``:
        - ``matrix``: a ``(T, R)`` numeric grid. If the top-left cell is ``NaN``
          the first row is read as r-values and the first column as t-values.
        - ``long``: three columns ``t, r, K`` reshaped onto the implied grid.
      Higher-dimensional kernels cannot live unambiguously in a flat CSV and
      raise ``NotImplementedError`` (use ``.npz``/``.mat`` instead).

    Any remaining keyword arguments are treated as explicit coordinate
    overrides, e.g. ``load_any('k.csv', r=my_r, t=my_t)``.

    Returns the same ``Kernel``/``Kernel2D``/``Kernel3D``/``Kernel4D`` objects
    as :func:`load_kernel`.
    """
    ext = os.path.splitext(str(path))[1].lower()
    if ext in (".npz", ".npy"):
        return load_kernel(path)
    if ext == ".mat":
        return _load_mat(path, time_axis=time_axis, **axes_hints)
    if ext in (".csv", ".txt"):
        return _load_csv(path, layout=layout, **axes_hints)
    raise ValueError(
        f"Unsupported extension {ext!r}. load_any reads .npz, .npy, .mat, "
        f".csv and .txt."
    )


def _coord_hints(d, hints):
    """Merge explicit coordinate-vector overrides into the dict passed to
    _build(). Known keys: t, r, x, y, z, w. Unknown keys are ignored."""
    for key in ("t",) + _SPATIAL_KEYS:
        if hints.get(key) is not None:
            d[key] = np.asarray(hints[key], dtype=float).ravel()


def _load_mat(path, time_axis=None, **axes_hints):
    from scipy.io import loadmat

    md = loadmat(path)
    arrays = {k: v for k, v in md.items() if not k.startswith("__")}
    if not arrays:
        raise ValueError(f"{path}: no usable arrays in .mat file")
    if "K" in arrays:
        K = np.asarray(arrays["K"], dtype=float)
    else:
        name = max(arrays, key=lambda k: (np.ndim(arrays[k]), np.size(arrays[k])))
        K = np.asarray(arrays[name], dtype=float)

    d = {}
    for key in ("t",) + _SPATIAL_KEYS:
        if key in arrays:
            v = np.asarray(arrays[key], dtype=float).squeeze()
            d[key] = v.reshape(1) if v.ndim == 0 else v
    _coord_hints(d, axes_hints)  # explicit overrides win

    t = d.get("t")
    spatial_lens = {len(d[k]) for k in _SPATIAL_KEYS if k in d}
    K = _orient_time_axis(K, t, spatial_lens, time_axis, str(path))
    return _build(K, d)


def _orient_time_axis(K, t, spatial_lens, time_axis, src):
    """Return K with the time axis on axis 0, detecting MATLAB time-last
    storage from the coordinate-vector lengths. Emits a warning when the
    orientation is genuinely ambiguous."""
    if K.ndim < 2:
        return K  # let _build raise the clear ndim error
    if time_axis == "first":
        return K
    if time_axis == "last":
        return np.moveaxis(K, -1, 0)
    if time_axis not in (None, "auto"):
        raise ValueError("time_axis must be 'first', 'last', or None/'auto'")

    first, last = K.shape[0], K.shape[-1]
    nt = None if t is None else len(t)
    if nt is not None:
        first_is_t, last_is_t = (first == nt), (last == nt)
        if last_is_t and not first_is_t:
            warnings.warn(
                f"{src}: time appears to be the last axis (length {last} matches "
                f"len(t)); transposed to (T, ...). Pass time_axis='first' to keep "
                f"the array as-is.",
                stacklevel=3,
            )
            return np.moveaxis(K, -1, 0)
        if first_is_t and not last_is_t:
            return K
        if not first_is_t and not last_is_t:
            warnings.warn(
                f"{src}: neither the first ({first}) nor last ({last}) axis matches "
                f"len(t)={nt}; assuming time is axis 0. Pass time_axis='first'|'last' "
                f"to override.",
                stacklevel=3,
            )
            return K
        # first and last both equal len(t): square in time; lean on spatial hints
        warnings.warn(
            f"{src}: time axis is ambiguous (first and last both have length "
            f"{nt}); assuming time is axis 0. Pass time_axis='first'|'last' to "
            f"override.",
            stacklevel=3,
        )
        return K

    # no t vector: fall back to spatial evidence
    if spatial_lens:
        if first in spatial_lens and last not in spatial_lens:
            warnings.warn(
                f"{src}: no t vector; the first axis ({first}) matches a spatial "
                f"length and the last does not, so time is assumed to be last and "
                f"the array was transposed. Pass time_axis='first' to keep as-is.",
                stacklevel=3,
            )
            return np.moveaxis(K, -1, 0)
        if last in spatial_lens and first not in spatial_lens:
            return K
    warnings.warn(
        f"{src}: no t vector to locate the time axis; assuming time is axis 0. "
        f"Pass time_axis='first'|'last' if that is wrong.",
        stacklevel=3,
    )
    return K


def _load_csv(path, layout="auto", **axes_hints):
    raw = np.atleast_2d(np.genfromtxt(path, delimiter=",", dtype=float))
    if raw.ndim != 2:
        raise ValueError(f"{path}: expected a 2D CSV grid, got ndim={raw.ndim}")
    if layout == "auto":
        layout = "long" if raw.shape[1] == 3 else "matrix"
    if layout == "long":
        if raw.shape[1] != 3:
            raise NotImplementedError(
                f"{path}: a long-format CSV must have exactly 3 columns (t, r, K) "
                f"for a 1D kernel; got {raw.shape[1]} columns. Multi-dimensional "
                f"kernels cannot be represented unambiguously in a flat CSV -- use "
                f".npz or .mat."
            )
        return _csv_long(raw, axes_hints)
    if layout == "matrix":
        return _csv_matrix(raw, axes_hints)
    raise ValueError(f"layout must be 'auto', 'matrix' or 'long'; got {layout!r}")


def _csv_long(raw, hints):
    ct, cr, ck = raw[:, 0], raw[:, 1], raw[:, 2]
    t_vals, t_idx = np.unique(ct, return_inverse=True)
    r_vals, r_idx = np.unique(cr, return_inverse=True)
    if len(t_vals) * len(r_vals) != len(ck):
        raise ValueError(
            f"long-format CSV is not a complete (t, r) grid: {len(t_vals)} times "
            f"x {len(r_vals)} positions != {len(ck)} rows"
        )
    K = np.full((len(t_vals), len(r_vals)), np.nan)
    K[t_idx, r_idx] = ck
    if np.isnan(K).any():
        raise ValueError("long-format CSV has missing (t, r) cells")
    d = {"t": t_vals, "r": r_vals}
    _coord_hints(d, hints)
    return _build(K, d)


def _csv_matrix(raw, hints):
    d = {}
    M = raw
    if M.shape[0] >= 2 and M.shape[1] >= 2 and np.isnan(M[0, 0]):
        d["r"] = M[0, 1:]
        d["t"] = M[1:, 0]
        K = M[1:, 1:]
    else:
        K = M
    _coord_hints(d, hints)
    return _build(np.asarray(K, dtype=float), d)
