# Tutorial — kernel-viewer as the stateflow-kernel plugin (one cell)

```python
from stateflow.kernel_plugin import (KernelExtractor, TransportAxis,
                                      CoherenceAxis, kernel_regime_report)

fs = KernelExtractor().from_path("examples/xxz_Sx_integrable.npz")
tp = TransportAxis().compute(fs)     # continuous: tp.scalar = p (e.g. 1.41)
co = CoherenceAxis().compute(fs)     # native trajectory: E(t) + episodes
print("transport p =", round(tp.scalar, 2), "->", TransportAxis().discretize(tp))
print("coherence ->", CoherenceAxis().discretize(co), "with", len(co.episodes), "episodes")
print("regime (optional, delegated, identical to kernel-viewer):",
      kernel_regime_report(fs).regime.value)
```

The regime read-out delegates to the unmodified `kernel_viewer.classify_any`,
so it is byte-for-byte identical to current kernel-viewer output.
