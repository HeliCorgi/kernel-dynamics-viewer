import numpy as np


def ring_strength(kernel):
    """E(t) = (max_{rho>0} P(rho,t) - P(0,t)) / |P(0,t)| and the position
    rho*(t) of the off-origin maximum. E > 0 means the profile maximum sits
    away from the source (a ring). Frames where |P(0,t)| has decayed below
    3x the noise floor are returned as NaN: the ratio is undefined there
    (e.g. a front kernel whose source-site value underflows)."""
    rho, P = kernel.symmetrized()
    floor = kernel.noise_floor()
    E = np.empty(len(kernel.t))
    rstar = np.empty(len(kernel.t))
    for i in range(len(kernel.t)):
        p0 = P[i, 0]
        off = P[i, 1:]
        if (not np.isfinite(p0) or abs(p0) < 3 * floor
                or not np.isfinite(off).any()):
            E[i] = np.nan; rstar[i] = np.nan
            continue
        j = int(np.nanargmax(off))
        E[i] = (off[j] - p0) / abs(p0)
        rstar[i] = rho[1 + j]
    return E, rstar


def ring_episodes(t, E, threshold=0.05):
    """Contiguous intervals where E(t) > threshold. Returns a list of
    (t_start, t_end, E_max)."""
    eps = []
    inside = False
    for i in range(len(t)):
        v = E[i]
        if np.isfinite(v) and v > threshold:
            if not inside:
                inside = True; start = t[i]; emax = v
            else:
                emax = max(emax, v)
        else:
            if inside:
                eps.append((start, t[i - 1], emax)); inside = False
    if inside:
        eps.append((start, t[-1], emax))
    return eps


def classify_coherence(t, E, threshold=0.05):
    """MONOTONE (no episode) / RING_TRANSIENT (one episode that ends) /
    RING_PERSISTENT (one episode still open at the last evaluable time) /
    RING_RECURRENT (>=2 episodes). Trailing NaN frames (e.g. masked
    underflowed source values) do not close an episode."""
    eps = ring_episodes(t, E, threshold)
    if len(eps) == 0:
        return "MONOTONE", eps
    if len(eps) >= 2:
        return "RING_RECURRENT", eps
    finite = np.asarray(t)[np.isfinite(E)]
    t_last = finite[-1] if len(finite) else t[-1]
    start, end, _ = eps[0]
    return ("RING_PERSISTENT" if np.isclose(end, t_last) else "RING_TRANSIENT"), eps
