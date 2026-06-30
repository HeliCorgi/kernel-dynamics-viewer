import numpy as np


def cosine_transform_r(rho, P, kvals):
    """C(k) = P(0) + 2 sum_{m>0} P(m) cos(k m) for symmetrized profiles."""
    out = np.empty((P.shape[0], len(kvals)))
    Pz = np.nan_to_num(P)
    for i in range(P.shape[0]):
        out[i] = Pz[i, 0] + 2 * (Pz[i, 1:][None, :] * np.cos(np.outer(kvals, rho[1:]))).sum(axis=1)
    return out


def cosine_transform_t(t, Ck, omegas, window="hann"):
    """S(omega) = dt * [C(0) w0 + 2 sum_n C(t_n) w_n cos(omega t_n)].
    Assumes C(-t) = C(t) (stationary autocorrelation, real Hamiltonian)."""
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    n = len(t)
    w = 0.5 * (1 + np.cos(np.pi * np.arange(n) / max(n - 1, 1))) if window == "hann" else np.ones(n)
    S = np.empty((len(omegas), Ck.shape[1]))
    for a, om in enumerate(omegas):
        cosv = np.cos(om * t) * w
        S[a] = dt * (Ck[0] * w[0] + 2 * (Ck[1:] * cosv[1:, None]).sum(axis=0))
    return S


def local_loglog_slope(t, y):
    """Centered local slope d log y / d log t; returns (t_mid, slope)."""
    t = np.asarray(t, dtype=float); y = np.asarray(y, dtype=float)
    m = (t > 0) & np.isfinite(y) & (y > 0)
    t, y = t[m], y[m]
    if len(t) < 3:
        return np.array([]), np.array([])
    lt, ly = np.log(t), np.log(y)
    sl = (ly[2:] - ly[:-2]) / (lt[2:] - lt[:-2])
    return t[1:-1], sl
