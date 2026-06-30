"""stateflow CLI -- command shape mirrors kernel-viewer.

    stateflow analyze --extractor kernel data.npz
    stateflow analyze --extractor eeg            (synthetic; no file needed)
    stateflow independence --extractor kernel
"""
from __future__ import annotations
import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _kernel_axes(path):
    from .kernel_plugin import KernelExtractor, TransportAxis, CoherenceAxis, kernel_regime_report
    fs = KernelExtractor().from_path(path)
    ta, ca = TransportAxis(), CoherenceAxis()
    return fs, ta, ca, ta.compute(fs), ca.compute(fs), kernel_regime_report(fs)


def main(argv=None):
    p = argparse.ArgumentParser(prog="stateflow",
                                description="Multi-axis trajectory analysis (continuous axes first)")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="compute axes + trajectories + optional regime")
    a.add_argument("--extractor", required=True, choices=["kernel", "eeg"])
    a.add_argument("data", nargs="?", default=None)
    a.add_argument("-o", "--out", default=None)
    iv = sub.add_parser("independence", help="print the three-part axis-independence evidence")
    iv.add_argument("--extractor", required=True, choices=["kernel"])

    args = p.parse_args(argv)

    if args.cmd == "independence":
        from .independence import kernel_axis_independence
        ev = kernel_axis_independence({
            "fixed_transport_a": "examples/xxz_Sx_integrable.npz",
            "fixed_transport_b": "examples/ising_q_chaotic.npz",
            "monotone_diffusive": "examples/xxz_Sz_diffusive.npz",
            "monotone_levy": "examples/randomwalk2d_mc.npz",
        })
        print(ev.summary())
        return 0

    if args.cmd == "analyze":
        from .viz import dashboard
        if args.extractor == "kernel":
            if not args.data:
                print("kernel extractor needs a data file"); return 2
            fs, ta, ca, tp, co, rep = _kernel_axes(args.data)
            print(f"transport axis (continuous): p = {tp.scalar:.3f}  -> {ta.discretize(tp)[0]}")
            print(f"coherence axis (continuous): E(t) trajectory, "
                  f"{len(co.episodes)} episode(s)  -> {ca.discretize(co)[0]}")
            print(f"regime read-out (optional, delegated to kernel-viewer): "
                  f"{rep.regime.value} / {rep.coherence}")
            out = args.out or "stateflow_kernel.png"
            # axis-space scatter over the bundled examples for context
            dashboard(tp, co, path=out)
            print(f"dashboard -> {out}")
        else:
            from .extractors.eeg import synth_eeg, SpectralSlopeAxis, CouplingAxis
            fs = synth_eeg()
            ss, cp = SpectralSlopeAxis(), CouplingAxis()
            st, ct = ss.compute(fs), cp.compute(fs)
            print(f"spectral_slope axis (continuous): mean exponent "
                  f"{np.nanmean(st.as_array()):.2f}")
            print(f"coupling axis (continuous): modulation-depth trajectory, "
                  f"max {np.nanmax(ct.as_array()):.2f}")
            from .core import TransitionAnalyzer
            tt = TransitionAnalyzer().from_threshold(ct.times, ct.as_array(), 0.3)
            for tr in tt.transitions:
                print(f"  coupling transition: {tr.kind} at t = {tr.time:.1f}s")
            out = args.out or "stateflow_eeg.png"
            dashboard(st, ct, path=out)
            print(f"dashboard -> {out}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
