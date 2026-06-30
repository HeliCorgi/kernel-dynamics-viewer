import numpy as np


def count_peaks(profile, prominence):
    """Count local maxima of a 1D profile with at least the given prominence.
    A peak at index i needs profile[i] to exceed the deepest valley separating
    it from a higher value by `prominence` (simple O(R^2) implementation)."""
    p = np.asarray(profile, dtype=float)
    n = len(p)
    peaks = []
    for i in range(n):
        left = p[i - 1] if i > 0 else -np.inf
        right = p[i + 1] if i < n - 1 else -np.inf
        if p[i] >= left and p[i] >= right and np.isfinite(p[i]):
            peaks.append(i)
    kept = 0
    for i in peaks:
        higher = [j for j in range(n) if p[j] > p[i]]
        if not higher:
            kept += 1
            continue
        prom = min(p[i] - p[min(i, j):max(i, j) + 1].min() for j in higher)
        if prom >= prominence:
            kept += 1
    return kept


def unimodality_score(kernel, prominence=None):
    """Per-time unimodality score = 1 / n_peaks of the symmetrized profile.

    1.0 = single peak; 0.5 = two peaks (ring), etc. Peaks are counted with a
    prominence threshold (default: 3 * kernel.noise_floor()) so that sampling
    noise does not register as structure."""
    rho, P = kernel.symmetrized()
    if prominence is None:
        prominence = 3.0 * kernel.noise_floor()
    scores = np.empty(len(kernel.t))
    npk = np.empty(len(kernel.t), dtype=int)
    for i in range(len(kernel.t)):
        n = count_peaks(P[i], prominence)
        npk[i] = n
        scores[i] = 1.0 / n if n > 0 else 0.0
    return scores, npk


def violation_amplitude(kernel):
    """V(t) = max_r [P(rho+1) - P(rho)]_+ / |P(0)|: amplitude of the strongest
    monotonicity violation of the symmetrized profile (0 = monotone). NaN
    where |P(0,t)| has decayed below 3x the noise floor (ratio undefined)."""
    rho, P = kernel.symmetrized()
    floor = kernel.noise_floor()
    out = np.empty(len(kernel.t))
    for i in range(len(kernel.t)):
        p0 = abs(P[i, 0])
        if not np.isfinite(p0) or p0 < 3 * floor:
            out[i] = np.nan
            continue
        diffs = np.diff(P[i][np.isfinite(P[i])])
        out[i] = max(float(diffs.max()), 0.0) / p0 if len(diffs) else 0.0
    return out
