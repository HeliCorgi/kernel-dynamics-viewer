"""stateflow tests: new abstractions, plugin equivalence, second domain,
and the non-negotiable constraint that kernel-viewer output is unchanged."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import warnings
warnings.filterwarnings("ignore")
import numpy as np


def test_feature_series_validation_and_mask():
    from stateflow.core import FeatureSeries
    fs = FeatureSeries(time=np.arange(5.0), features=np.random.rand(5, 2),
                       feature_names=["a", "b"])
    assert fs.n_features == 2 and fs.mask.all()
    assert fs.metadata["provenance"] == "unspecified"
    try:
        FeatureSeries(time=np.arange(5.0), features=np.random.rand(4, 2))
        assert False, "should reject mismatched lengths"
    except ValueError:
        pass


def test_trajectory_one_class_three_uses():
    from stateflow.core import Trajectory, Transition
    # scalar-per-step (axis), object-per-step (richer), transitions attached
    t1 = Trajectory(times=np.arange(3.0), values=np.array([1.0, 2.0, 3.0]))
    assert t1.as_array().shape == (3,)
    t2 = Trajectory(times=np.arange(2.0), values=[{"a": 1}, {"a": 2}])
    assert isinstance(t2.values, list)
    t3 = Trajectory(times=np.arange(2.0), values=np.zeros(2),
                    transitions=[Transition(1, 1.0, "x->y")])
    assert t3.transitions[0].kind == "x->y"


def test_transition_analyzer_matches_tracking_semantics():
    from stateflow.core import TransitionAnalyzer
    ta = TransitionAnalyzer()
    # trailing None must not emit a transition (tracking.py semantics)
    tr = ta.from_labels(np.arange(5.0), ["A", "A", "B", None, None])
    assert [t.kind for t in tr.transitions] == ["A->B"]
    th = ta.from_threshold(np.arange(5.0), np.array([0, 1, 1, 0, 1.0]), 0.5)
    assert [t.kind for t in th.transitions] == ["rise", "fall", "rise"]
    assert ta.recurrence(th) == 2  # two upward crossings


def test_kernel_axes_continuous_and_protocol():
    from stateflow.core import Axis
    from stateflow.kernel_plugin import KernelExtractor, TransportAxis, CoherenceAxis
    ex = KernelExtractor()
    base = os.path.join(os.path.dirname(__file__), "..", "examples")
    fs = ex.from_path(os.path.join(base, "xxz_Sx_integrable.npz"))
    ta, ca = TransportAxis(), CoherenceAxis()
    assert isinstance(ta, Axis) and isinstance(ca, Axis)
    tp = ta.compute(fs)
    assert np.isfinite(tp.scalar)                      # continuous value present
    assert ta.discretize(tp) == ["superdiffusive"]     # label is downstream
    co = ca.compute(fs)
    assert co.as_array().shape[0] == len(fs.time)      # E(t) is a trajectory
    assert ca.discretize(co) == ["RING_RECURRENT"]


def test_regime_readout_delegates_unchanged():
    """The optional regime read-out must equal kernel_viewer.classify_any."""
    from stateflow.kernel_plugin import KernelExtractor, kernel_regime_report
    from kernel_viewer.io.loader import load_kernel
    from kernel_viewer.regimes.classifier import classify_any
    base = os.path.join(os.path.dirname(__file__), "..", "examples")
    for f in ("xxz_Sx_integrable.npz", "xxz_Sz_diffusive.npz", "randomwalk2d_mc.npz"):
        path = os.path.join(base, f)
        direct = classify_any(load_kernel(path))
        viaplugin = kernel_regime_report(KernelExtractor().from_path(path))
        assert direct.regime == viaplugin.regime
        assert direct.coherence == viaplugin.coherence


def test_second_domain_eeg_runs_through_same_core():
    from stateflow.core import FeatureSeries, Axis, TransitionAnalyzer
    from stateflow.extractors.eeg import synth_eeg, SpectralSlopeAxis, CouplingAxis
    fs = synth_eeg()
    assert isinstance(fs, FeatureSeries)
    ss, cp = SpectralSlopeAxis(), CouplingAxis()
    assert isinstance(ss, Axis) and isinstance(cp, Axis)
    ct = cp.compute(fs)
    # PAC switches on at 10s -> a rise transition after onset
    tt = TransitionAnalyzer().from_threshold(ct.times, ct.as_array(), 0.3)
    rises = [t for t in tt.transitions if t.kind == "rise"]
    assert rises and rises[0].time >= 9.0


def test_independence_three_parts_no_correlation_criterion():
    from stateflow.independence import kernel_axis_independence
    base = os.path.join(os.path.dirname(__file__), "..", "examples")
    ev = kernel_axis_independence({
        "fixed_transport_a": os.path.join(base, "xxz_Sx_integrable.npz"),
        "fixed_transport_b": os.path.join(base, "ising_q_chaotic.npz"),
        "monotone_diffusive": os.path.join(base, "xxz_Sz_diffusive.npz"),
        "monotone_levy": os.path.join(base, "randomwalk2d_mc.npz"),
    })
    assert ev.computational["data_flow"].startswith("none")
    assert len(ev.distinct_information) >= 2     # counterexamples, both directions
    assert len(ev.failure_mode) >= 2
    # correlation is reported but is NOT the criterion: evidence holds even
    # though r can be high.
    assert ev.correlation_diagnostic is not None


def test_visualization_renders():
    import matplotlib
    matplotlib.use("Agg")
    from stateflow.kernel_plugin import KernelExtractor, TransportAxis, CoherenceAxis
    from stateflow.viz import dashboard
    ex = KernelExtractor()
    base = os.path.join(os.path.dirname(__file__), "..", "examples")
    fs = ex.from_path(os.path.join(base, "xxz_Sx_integrable.npz"))
    fig = dashboard(TransportAxis().compute(fs), CoherenceAxis().compute(fs))
    assert fig is not None


if __name__ == "__main__":
    test_feature_series_validation_and_mask()
    test_trajectory_one_class_three_uses()
    test_transition_analyzer_matches_tracking_semantics()
    test_kernel_axes_continuous_and_protocol()
    test_regime_readout_delegates_unchanged()
    test_second_domain_eeg_runs_through_same_core()
    test_independence_three_parts_no_correlation_criterion()
    test_visualization_renders()
    print("ALL STATEFLOW TESTS PASS")
