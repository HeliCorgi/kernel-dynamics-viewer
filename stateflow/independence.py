"""Axis-independence evidence (the three-part check), generalized from the
counterexample style already used in ``kernel_viewer.classify``'s docstring.

This module NEVER uses a correlation threshold as the criterion. It returns
structured evidence in three parts:

  1. computational_independence -- a statement that the two axes are computed
     by different algorithms with no data flow (recorded provenance strings);
  2. distinct_information -- explicit counterexample pairs where axis 1 is
     fixed while axis 2 varies, and vice versa (the caller supplies the cases);
  3. failure_mode_independence -- a case where one axis is unreliable while the
     other stays reliable.

Pearson correlation MAY be reported as a descriptive diagnostic, clearly
labelled as not-the-criterion.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any
import numpy as np

from .core import FeatureSeries


@dataclass
class IndependenceEvidence:
    axis1: str
    axis2: str
    computational: dict[str, Any]
    distinct_information: list[dict[str, Any]] = field(default_factory=list)
    failure_mode: list[dict[str, Any]] = field(default_factory=list)
    correlation_diagnostic: float | None = None

    def summary(self) -> str:
        lines = [f"Axis independence: {self.axis1}  vs  {self.axis2}", ""]
        lines.append("1) Computational independence:")
        lines.append(f"   {self.axis1}: {self.computational[self.axis1]}")
        lines.append(f"   {self.axis2}: {self.computational[self.axis2]}")
        lines.append(f"   data flow between axes: {self.computational['data_flow']}")
        lines.append("")
        lines.append("2) Distinct-information counterexamples:")
        for c in self.distinct_information:
            lines.append(f"   - {c['description']}: "
                         f"{self.axis1}={c['axis1_value']}, {self.axis2}={c['axis2_value']}")
        lines.append("")
        lines.append("3) Failure-mode independence:")
        for c in self.failure_mode:
            lines.append(f"   - {c['description']}")
        if self.correlation_diagnostic is not None:
            lines.append("")
            lines.append(f"(diagnostic only, NOT the criterion) Pearson r over a "
                         f"sample = {self.correlation_diagnostic:+.3f}")
        return "\n".join(lines)


def kernel_axis_independence(example_paths: dict[str, str]) -> IndependenceEvidence:
    """Build the three-part evidence for the kernel plugin's transport vs.
    coherence axes from bundled example kernels.

    ``example_paths`` should map roles to file paths, e.g.
    {"ballistic_monotone": ..., "ballistic_recurrent": ...,
     "fixed_transport_a": ..., "fixed_transport_b": ...}.
    """
    from .kernel_plugin import KernelExtractor, TransportAxis, CoherenceAxis
    ex = KernelExtractor(); ta = TransportAxis(); ca = CoherenceAxis()

    def pt(path):
        fs = ex.from_path(path)
        tp = ta.compute(fs); co = ca.compute(fs)
        return round(tp.scalar, 2), ca.discretize(co)[0], ta.discretize(tp)[0]

    dist = []
    # Case A: fix transport ~ similar p, vary coherence (integrable vs diffusive)
    if "fixed_transport_a" in example_paths and "fixed_transport_b" in example_paths:
        pa, ca_lab, ta_lab = pt(example_paths["fixed_transport_a"])
        pb, cb_lab, tb_lab = pt(example_paths["fixed_transport_b"])
        dist.append({"description": "two kernels with overlapping transport exponent "
                                    "but different coherence class",
                     "axis1_value": f"p~{pa} vs p~{pb} (transport: {ta_lab}/{tb_lab})",
                     "axis2_value": f"{ca_lab} vs {cb_lab}"})
    # Case B: vary transport, hold coherence MONOTONE (diffusive vs levy)
    if "monotone_diffusive" in example_paths and "monotone_levy" in example_paths:
        pa, ca_lab, ta_lab = pt(example_paths["monotone_diffusive"])
        pb, cb_lab, tb_lab = pt(example_paths["monotone_levy"])
        dist.append({"description": "two kernels both MONOTONE but different transport",
                     "axis1_value": f"p~{pa} ({ta_lab}) vs p~{pb} ({tb_lab})",
                     "axis2_value": f"{ca_lab} vs {cb_lab}"})

    fail = [
        {"description": "K(0,t) oscillating/underflowing (e.g. a wave node) makes "
                        "E(t) spike or go undefined -> coherence confidence drops to "
                        "0 on those frames, while sigma^2(t) (a full-field second "
                        "moment) is unaffected -> transport stays reliable."},
        {"description": "A heavy power-law spatial tail makes sigma^2 window-limited "
                        "-> transport exponent biased toward 1, while E(t) (a "
                        "source-vs-off-source contrast) still correctly reports the "
                        "monotone profile -> coherence stays reliable."},
    ]
    comp = {
        "transport": "second moment sigma^2(t) of |K| + log-log slope (transport.py)",
        "coherence": "ring-strength E(t) = (max off-source - K(0))/|K(0)| (ring.py)",
        "data_flow": "none (separate functions, separate inputs from the kernel)",
    }
    # diagnostic-only correlation across the supplied examples (NOT the criterion)
    corr = None
    pts = [pt(p) for p in example_paths.values()]
    ps = np.array([a for a, _, _ in pts], dtype=float)
    # encode coherence as an ordinal just for the descriptive diagnostic
    order = {"MONOTONE": 0, "RING_TRANSIENT": 1, "RING_PERSISTENT": 2, "RING_RECURRENT": 3}
    cs = np.array([order.get(c, np.nan) for _, c, _ in pts], dtype=float)
    good = np.isfinite(ps) & np.isfinite(cs)
    if good.sum() >= 3 and np.std(ps[good]) > 0 and np.std(cs[good]) > 0:
        corr = float(np.corrcoef(ps[good], cs[good])[0, 1])
    return IndependenceEvidence("transport", "coherence", comp,
                                distinct_information=dist, failure_mode=fail,
                                correlation_diagnostic=corr)
