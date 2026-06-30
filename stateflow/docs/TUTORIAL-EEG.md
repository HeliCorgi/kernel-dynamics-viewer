# Tutorial — synthetic EEG, the second domain (one cell)

```python
from stateflow.extractors.eeg import synth_eeg, SpectralSlopeAxis, CouplingAxis
from stateflow.core import TransitionAnalyzer

fs = synth_eeg(couple_onset_s=10.0)            # 1/f signal; PAC switches on at 10s
slope = SpectralSlopeAxis().compute(fs)        # continuous 1/f exponent trajectory
coupl = CouplingAxis().compute(fs)             # continuous modulation-depth trajectory

tt = TransitionAnalyzer().from_threshold(coupl.times, coupl.as_array(), 0.3)
print("coupling onset transitions:", [(round(t.time,1), t.kind) for t in tt.transitions])
```

The *same* core (`FeatureSeries`, `Axis`, `Trajectory`, `TransitionAnalyzer`)
runs unchanged on a second domain — no kernel concepts involved.
