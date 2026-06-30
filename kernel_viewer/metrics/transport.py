import numpy as np
from ..core.transforms import local_loglog_slope


def transport_exponent(kernel, fit_window=(0.5, 1.0), edge_limit=0.3):
    """Late-window exponent p in sigma^2 ~ t^p, with the boundary-contaminated
    tail (edge_fraction > edge_limit) excluded.

    fit_window = (a, b): use the fraction [a, b] of the clean time range.
    Returns (p, t_used, local_slopes, t_clean_max)."""
    t = kernel.t
    s2 = kernel.sigma2()
    ef = kernel.edge_fraction()
    clean = np.isfinite(ef) & (ef <= edge_limit)
    t_clean_max = t[clean][-1] if clean.any() else t[-1]
    tm, sl = local_loglog_slope(t, s2)
    m = (tm <= t_clean_max) & (tm >= fit_window[0] * t_clean_max) & (tm <= fit_window[1] * t_clean_max)
    if not m.any():
        m = tm <= t_clean_max
    if not m.any():
        return np.nan, tm, sl, t_clean_max
    return float(np.mean(sl[m])), tm, sl, t_clean_max


def tail_shape(kernel, at_time=None, edge_limit=0.3):
    """Fit -log(P(rho)/P(0)) at the latest clean time against three forms:
    a*rho^2 (gaussian), a*rho (exponential), a*log(1+rho) (power-law tail).
    Returns (best_label, r2_dict, t_used)."""
    rho, P = kernel.symmetrized()
    ef = kernel.edge_fraction()
    floor = kernel.noise_floor()
    idx = None
    for i in range(len(kernel.t) - 1, -1, -1):
        if np.isfinite(ef[i]) and ef[i] <= edge_limit and P[i, 0] > 3 * floor:
            idx = i
            break
    if idx is None or (at_time is not None):
        idx = int(np.argmin(np.abs(kernel.t - at_time))) if at_time is not None else len(kernel.t) - 1
    p = P[idx]
    m = np.isfinite(p) & (p > 3 * floor) & (rho >= 1)
    rr, pp = rho[m], p[m]
    if len(rr) < 6 or not np.isfinite(P[idx, 0]) or P[idx, 0] <= 0:
        return "UNKNOWN", {}, float(kernel.t[idx])
    # discriminate on the OUTER HALF of the usable range, where the three
    # decay laws separate (mid-range least squares cannot tell them apart)
    cut = rr[len(rr) // 2]
    sel = rr >= cut
    x, y = rr[sel], np.log(pp[sel])
    if len(x) < 3:
        return "UNKNOWN", {}, float(kernel.t[idx])
    # power-law test by log-log curvature: a power-law tail is a straight
    # line in (log r, log P) (curvature ~ 0); gaussian/exponential tails are
    # strongly concave there (|c| >> 1 in the tested models).
    lx, ly = np.log(x), y
    r2 = {}
    if len(lx) >= 4:
        c, b, _ = np.polyfit(lx, ly, 2)
        r2["loglog_curvature"] = round(float(c), 3)
        r2["loglog_slope"] = round(float(b), 3)
        if abs(c) < 2.0 and -6.0 < b < -0.5:
            return "powerlaw", r2, float(kernel.t[idx])
    for label, f in (("gaussian", x ** 2), ("exponential", x)):
        A = np.vstack([f, np.ones_like(f)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        ss = ((y - y.mean()) ** 2).sum()
        r2[label] = 1 - ((y - pred) ** 2).sum() / ss if ss > 0 else np.nan
    pair = {k: r2[k] for k in ("gaussian", "exponential") if k in r2}
    best = max(pair, key=lambda k: (pair[k] if np.isfinite(pair[k]) else -np.inf))
    return best, r2, float(kernel.t[idx])
