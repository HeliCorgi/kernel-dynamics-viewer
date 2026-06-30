# stateflow — Honest Limitations

## Per-axis (kernel plugin)
- **TransportAxis**: the exponent is the exponent *of the time window*; short
  windows report transients (the bundled XXZ data is a documented example).
  Heavy power-law tails make σ² window-limited — the delegated regime read-out
  promotes Lévy separately, but the bare exponent can still read ≈1.
- **CoherenceAxis**: E(t) is normalized by K(0,t); at wave nodes K(0,t)→0 and
  E spikes or is undefined (masked to NaN). Read the **episode structure**, not
  the spike magnitudes. In 2D+ the axis runs on the radial reduction, so it
  inherits the inscribed-circle/sphere range limits and the isotropy caveats.

## Per-axis (EEG plugin, synthetic)
- **SpectralSlopeAxis**: a single broadband log-log fit; mixed white+pink
  content lowers R² (reported as confidence). Not a substitute for FOOOF-style
  aperiodic/periodic separation.
- **CouplingAxis**: a modulation-depth estimate over short windows; sensitive
  to filter band choice and window length. It is a *demonstration* that the
  same core handles a second domain, not a validated neuroscience tool.

## Core
- ≤3 axes (visualization and interpretation break down beyond that).
- Axes are **user-designed**; nothing is learned or auto-selected.
- No hidden-state inference, no fitting, no optimization in core.
- Independence is argued by evidence (computational + distinct-information +
  failure-mode), **never** by a correlation threshold; correlation is reported
  only as a descriptive diagnostic and can be high even when axes are
  independent.
