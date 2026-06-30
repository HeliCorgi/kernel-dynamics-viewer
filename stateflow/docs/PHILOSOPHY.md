# stateflow — Philosophy

The center of stateflow is **the evolution of measurements**, not the regime:

```
Feature -> Axis value (continuous) -> Trajectory -> Transition -> (optional) Regime
```

A regime is the last and optional step: a human-readable label placed on a
*location in axis space*. The library is healthy even if a user never asks for
a label — the trajectories and transitions stand on their own.

## What kernel-viewer taught us (preserved here)

1. **Don't infer transport from peak counts.** Early ballistic kernels are
   unimodal too — measure σ²(t) slope directly. (`TransportAxis` does exactly
   this, via the unmodified `kernel_viewer.transport_exponent`.)
2. **Don't assume diffusion from second-moment scaling.** Heavy tails make σ²
   window-limited and fake p≈1 — detect tails separately. (Preserved in the
   delegated regime read-out.)
3. **Don't assume axes are correlated.** Ballistic systems can be monotone *or*
   ring-recurrent — compute axes independently and keep both.

stateflow is an **articulation** of the design already implicit in
kernel-viewer, not a new framework. The two new abstractions (`FeatureSeries`,
`Trajectory`) are each evidenced by existing code; `Axis` is a Protocol; the
transition analyzer generalizes `tracking.py`.
