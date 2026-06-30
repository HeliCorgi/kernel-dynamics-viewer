from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from ..metrics.transport import transport_exponent, tail_shape
from ..metrics.ring import ring_strength, classify_coherence
from ..metrics.unimodality import unimodality_score, violation_amplitude
from ..core.spectral import compute_spectrum
from ..core.kernel2d import Kernel2D
from ..core.kernel3d import Kernel3D
from ..core.kernel4d import Kernel4D


class Regime(Enum):
    DIFFUSIVE = "diffusive"          # sigma^2 ~ t^1
    SUPERDIFFUSIVE = "superdiffusive"  # 1 < p < 1.8
    BALLISTIC = "ballistic"          # sigma^2 ~ t^2
    SUBDIFFUSIVE = "subdiffusive"    # p < 0.8
    LEVY = "levy"                    # superdiffusive + power-law spatial tail
    UNKNOWN = "unknown"


@dataclass
class RegimeReport:
    regime: Regime
    coherence: str                  # MONOTONE / RING_TRANSIENT / RING_PERSISTENT / RING_RECURRENT
    exponent: float                 # p in sigma^2 ~ t^p
    features: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def summary(self):
        lines = [
            f"regime      : {self.regime.value}",
            f"coherence   : {self.coherence}",
            f"exponent p  : {self.exponent:.2f}  (sigma^2 ~ t^p; 1=diffusive, 2=ballistic)",
        ]
        for k, v in self.features.items():
            lines.append(f"{k:<12}: {v}")
        for n in self.notes:
            lines.append(f"note        : {n}")
        return "\n".join(lines)


def classify(kernel, ring_threshold=0.05, edge_limit=0.3):
    """Two-axis classification of a kernel.

    Axis 1 (transport): exponent p of sigma^2 ~ t^p in the boundary-clean
    window, refined by the spatial tail shape (power-law tail + p > 1 -> LEVY).
    Axis 2 (coherence): ring structure of the profile over time
    (MONOTONE / RING_TRANSIENT / RING_PERSISTENT / RING_RECURRENT).

    These axes are independent: ballistic kernels can be unimodal and
    diffusive systems can show transient rings. Chaotic vs integrable
    typically shows up on axis 2 (transient vs recurrent rings), not axis 1."""
    notes = []
    p, tm, sl, t_clean = transport_exponent(kernel, edge_limit=edge_limit)
    E, rstar = ring_strength(kernel)
    coh, episodes = classify_coherence(kernel.t, E, ring_threshold)
    scores, npk = unimodality_score(kernel)
    V = violation_amplitude(kernel)
    tail, r2, t_tail = tail_shape(kernel, edge_limit=edge_limit)

    if not np.isfinite(p):
        regime = Regime.UNKNOWN
        notes.append("could not determine a transport exponent (too few clean points)")
    elif p >= 1.8:
        regime = Regime.BALLISTIC
    elif p > 1.2:
        regime = Regime.LEVY if tail == "powerlaw" else Regime.SUPERDIFFUSIVE
    elif p >= 0.8:
        regime = Regime.DIFFUSIVE
    else:
        regime = Regime.SUBDIFFUSIVE

    # Heavy-tail promotion: for power-law spatial tails the second moment is
    # dominated by the finite window and fakes p ~ 1, so sigma^2 alone cannot
    # see a Levy kernel. Promote when the power-law tail wins clearly.
    if regime in (Regime.DIFFUSIVE, Regime.SUPERDIFFUSIVE) and tail == "powerlaw":
        regime = Regime.LEVY
        notes.append("promoted to LEVY by power-law spatial tail (sigma^2 is window-limited for heavy tails)")

    if t_clean < kernel.t[-1]:
        notes.append(f"boundary contamination beyond t={t_clean:g} (edge_fraction > {edge_limit}); tail excluded")
    tot = kernel.total()
    if np.nanstd(tot) < 0.02 * abs(np.nanmean(tot)) and abs(np.nanmean(tot)) > 0:
        notes.append("kernel weight conserved (sum rule constant to <2%)")
    else:
        notes.append("kernel weight NOT conserved (decaying channel)")

    feats = {
        "dimensionality": 1,
        "unimodality": f"mean {np.nanmean(scores):.2f}, min {np.nanmin(scores):.2f}",
        "violation V": f"max {np.nanmax(V):.3f} (relative to K0)",
        "ring": f"{len(episodes)} episode(s), max E={max((e[2] for e in episodes), default=0):.2f}",
        "tail shape": f"{tail} (R2: " + ", ".join(f"{k}={v:.2f}" for k, v in r2.items()) + f") at t={t_tail:g}",
        "clean window": f"t <= {t_clean:g}",
    }
    return RegimeReport(regime=regime, coherence=coh, exponent=p, features=feats, notes=notes)


def classify2d(kernel2d, ring_threshold=0.05, edge_limit=0.3, aniso_warn=0.10):
    """Classify a 2D kernel: radial reduction + the 1D two-axis machinery,
    with full-2D moments and 2D anisotropy features added.

    Coherence reading differs in 2D: a propagating mode's wavefront IS a
    circle, so RING_* on the radial profile means "a coherent propagating
    front is present", and is generic for wave-like transport (it does not
    by itself indicate integrability)."""
    rk = kernel2d.radial()
    rep = classify(rk, ring_threshold=ring_threshold, edge_limit=edge_limit)
    A = kernel2d.anisotropy()
    harm = kernel2d.angular_harmonics()
    ef = kernel2d.edge_fraction()
    clean = np.where(np.isfinite(ef) & (ef <= edge_limit))[0]
    late = clean[len(clean) // 2:] if len(clean) >= 2 else np.arange(len(kernel2d.t))
    a_late = float(np.nanmax(A[late])) if np.isfinite(A[late]).any() else float("nan")
    c2l = float(np.nanmedian(harm[2][late])); c4l = float(np.nanmedian(harm[4][late]))
    rep.features["dimensionality"] = 2
    rep.features["isotropy_score"] = round(float(np.clip(1.0 - max(a_late, c4l), 0, 1)), 3)
    rep.features["anisotropy"] = f"A {a_late:.3f}; c2 {c2l:.3f}, c4 {c4l:.3f} (late clean window, r>=2)"
    if np.nanmax([a_late, c4l]) > aniso_warn:
        rep.notes.append("anisotropic kernel: radial averaging mixes directions; "
                         "ring/monotonicity calls on the radial profile may be smearing artifacts")
    rep.notes.append("2D input: transport moments computed on the full 2D kernel; "
                     "shape metrics on the radial profile (inscribed circle only)")
    return rep


def classify_any(kernel, **kw):
    """Dispatch by kernel type: Kernel4D/3D/2D -> classify4d/3d/2d, else the
    1D classify."""
    if isinstance(kernel, Kernel4D):
        return classify4d(kernel, **kw)
    if isinstance(kernel, Kernel3D):
        return classify3d(kernel, **kw)
    if isinstance(kernel, Kernel2D):
        return classify2d(kernel, **kw)
    return classify(kernel, **kw)


def classify3d(kernel3d, ring_threshold=0.05, edge_limit=0.3, aniso_warn=0.10):
    """Classify a 3D kernel: spherical radial reduction + the 1D two-axis
    machinery with full-3D moments, plus FA (covariance) and Q4 (cubic)
    anisotropy. As in 2D, a propagating mode's front is a spherical shell, so
    RING_* means "a coherent propagating front is present"."""
    rk = kernel3d.radial()
    rep = classify(rk, ring_threshold=ring_threshold, edge_limit=edge_limit)
    FA = kernel3d.fractional_anisotropy()
    Q4 = kernel3d.cubic_anisotropy()
    ef = kernel3d.edge_fraction()
    clean = np.where(np.isfinite(ef) & (ef <= edge_limit))[0]
    late = clean[len(clean) // 2:] if len(clean) >= 2 else np.arange(len(kernel3d.t))
    fa_l = float(np.nanmax(FA[late])) if np.isfinite(FA[late]).any() else float("nan")
    q4_l = float(np.nanmedian(Q4[late]))
    rep.features["dimensionality"] = 3
    rep.features["isotropy_score"] = round(float(np.clip(1.0 - max(fa_l, abs(q4_l) / 0.4), 0, 1)), 3)
    rep.features["anisotropy"] = f"FA {fa_l:.3f}; Q4 {q4_l:+.3f} (late clean window, r>=2; +0.4 = axis-aligned)"
    if max(fa_l, abs(q4_l) / 0.4) > aniso_warn:
        rep.notes.append("anisotropic kernel: spherical averaging mixes directions; "
                         "ring/monotonicity calls on the radial profile may be smearing artifacts")
    rep.notes.append("3D input: transport moments computed on the full 3D kernel; "
                     "shape metrics on the spherical-shell profile (inscribed sphere only)")
    return rep


def classify4d(kernel4d, ring_threshold=0.05, edge_limit=0.3, aniso_warn=0.10):
    """Classify a 4D kernel: hyperspherical radial reduction + the 1D two-axis
    machinery with full-4D moments, plus generalized FA and Q4 (isotropic
    constant 3/(d+2) = 1/2; +0.5 = axis-aligned). RING_* means "a coherent
    propagating front is present" -- in d=4 the front's surface dilution is
    1/r^3, so front kernels lose their source-site value very fast and clean
    windows are short."""
    rk = kernel4d.radial()
    rep = classify(rk, ring_threshold=ring_threshold, edge_limit=edge_limit)
    FA = kernel4d.fractional_anisotropy()
    Q4 = kernel4d.cubic_anisotropy()
    ef = kernel4d.edge_fraction()
    clean = np.where(np.isfinite(ef) & (ef <= edge_limit))[0]
    late = clean[len(clean) // 2:] if len(clean) >= 2 else np.arange(len(kernel4d.t))
    fa_l = float(np.nanmax(FA[late])) if np.isfinite(FA[late]).any() else float("nan")
    q4_l = float(np.nanmedian(Q4[late]))
    rep.features["dimensionality"] = 4
    rep.features["isotropy_score"] = round(float(np.clip(1.0 - max(fa_l, abs(q4_l) / 0.5), 0, 1)), 3)
    rep.features["anisotropy"] = f"FA {fa_l:.3f}; Q4 {q4_l:+.3f} (late clean window, r>=2; +0.5 = axis-aligned)"
    if max(fa_l, abs(q4_l) / 0.5) > aniso_warn:
        rep.notes.append("anisotropic kernel: hyperspherical averaging mixes directions; "
                         "ring/monotonicity calls on the radial profile may be smearing artifacts")
    rep.notes.append("4D input: transport moments computed on the full 4D kernel; "
                     "shape metrics on the hyperspherical-shell profile (inscribed hypersphere only)")
    return rep
