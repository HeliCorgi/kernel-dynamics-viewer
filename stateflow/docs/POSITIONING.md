# stateflow — Positioning

The novelty is **design discipline**, not a new algorithm.

Compatible with, but **out of scope for the core** (these may *consume* a
stateflow `FeatureSeries` / axis trajectory / regime trajectory; stateflow does
not reimplement or compete with them):

- Hidden Markov Models (`hmmlearn`) — stateflow infers **no hidden state**.
- Change-point detection (`ruptures`) — stateflow reports transitions on
  explicit, continuous axes, with no fitting.
- Clustering / regime-switching / state-space filters (`scikit-learn`,
  `statsmodels`).
- Time-series feature sets (catch22, tsfresh, hctsa) — these extract *many*
  features and deliberately **decorrelate** them to feed a classifier.
  stateflow does the opposite: a *small* set of axes, **independent by
  construction**, whose **trajectories and transitions are the object of
  study**.

The one real differentiator is the **combination**: a small set of
independent-by-construction, continuous-valued-first, interpretable axes,
tracked as trajectories with transitions, with documented failure modes, using
NumPy + SciPy + Matplotlib only, and **no black boxes**.
