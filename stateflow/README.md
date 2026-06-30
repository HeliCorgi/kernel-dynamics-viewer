# stateflow

A **trajectory library** for multi-axis regime analysis — a generalization of
[`kernel-viewer`](../README.md)'s philosophy to arbitrary dynamical systems.

> One-line identity: stateflow analyzes **trajectories of measurements in an
> explicitly designed axis space.** The core object is the *evolution of
> measurements*, not the regime.

```python
from stateflow.kernel_plugin import KernelExtractor, TransportAxis, CoherenceAxis
fs = KernelExtractor().from_path("examples/xxz_Sx_integrable.npz")
p  = TransportAxis().compute(fs).scalar     # continuous exponent (e.g. 1.41)
E  = CoherenceAxis().compute(fs)            # native E(t) trajectory + episodes
```

```bash
stateflow analyze --extractor kernel examples/xxz_Sx_integrable.npz
stateflow analyze --extractor eeg            # synthetic second domain, no file
stateflow independence --extractor kernel    # three-part axis-independence evidence
```

## Design in one breath
- Two new abstractions only: **`FeatureSeries`** (container) and **`Trajectory`**
  (one class, three uses). `Axis` is a `typing.Protocol`; the transition
  analyzer generalizes `kernel_viewer/metrics/tracking.py`.
- **Continuous axis values first**; labels are optional, downstream, lossless.
- **Regime is terminal and optional** — a location in axis space.
- **Axis independence** is shown by evidence (computational + distinct-info
  counterexamples + failure-mode), **never** a correlation threshold.
- **No** HMM / clustering / NN / optimizer in core. **NumPy + SciPy +
  Matplotlib only.**
- **kernel-viewer is unchanged**: stateflow wraps it; all existing tests pass
  with identical output and the regime read-out delegates to `classify_any`.

See `stateflow/docs/` for philosophy, positioning, tutorials, and per-axis
limitations.
