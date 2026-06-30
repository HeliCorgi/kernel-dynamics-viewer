"""Generate a trajectory-ensemble example kernel with yupi (a cross-library check).

This script is NOT part of the package and is the ONLY place yupi is imported.
It synthesises an ensemble of 2D Brownian motion with yupi -- an independent
implementation -- converts it into a kernel via the pure-NumPy helper in
``kernel_viewer.io.trajectories``, classifies it with the unmodified
``classify_any``, and saves the npz bundled in examples/.

An ensemble of walkers all starting at the origin IS the propagator of the
process, so histogramming their positions at each time yields K(x, y, t). The
scientific point: a correct classification of someone *else's* process is a
cross-library consistency check, not a tautology.

Heavy-tailed (Levy) processes are intentionally OUT OF SCOPE for this
cross-check: a square-lattice kernel clips the heavy tail, and the tail-shape
heuristic is neither robust nor fast enough for them -- use dedicated
anomalous-diffusion tools (MSD-exponent fits, van Hove functions,
scipy.stats.levy_stable, yupi's own analysis) instead.

Run:  pip install -e ".[examples]"  &&  python examples/generate_yupi_examples.py

Note: the committed npz is the source of truth for the tests; regeneration is a
convenience / yupi-API-drift check, not a reproducibility guarantee.
"""
import os

import numpy as np
from yupi.generators import RandomWalkGenerator

from kernel_viewer.io.trajectories import ensemble_to_kernel2d
from kernel_viewer.regimes.classifier import classify_any

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 12345
# Simple unbiased +/-1 lattice step per dimension: P(-) = P(+) = 0.5, P(stay) = 0.
ACTIONS_PROB = np.array([[0.5, 0.0, 0.5], [0.5, 0.0, 0.5]])


def _stack(trajs):
    """yupi trajectories -> (N_walkers, T_steps, 2) position array."""
    return np.stack([np.asarray(tr.r, dtype=float) for tr in trajs], axis=0)


def make_random_walk():
    """Independent 2D Brownian motion -> lattice Kernel2D (light-tailed)."""
    gen = RandomWalkGenerator(T=12.0, dim=2, dt=0.5, actions_prob=ACTIONS_PROB,
                              seed=SEED)
    positions = _stack(gen.generate(8000))
    return ensemble_to_kernel2d(positions, L=41)


def main():
    rw = make_random_walk()
    rw_path = os.path.join(HERE, "yupi_randomwalk2d.npz")
    np.savez_compressed(rw_path, x=rw.x, y=rw.y, t=rw.t, K=rw.K.astype(np.float32))
    rep_rw = classify_any(rw)
    print("yupi random walk (lattice Kernel2D):")
    print(rep_rw.summary())
    print(f"-> {os.path.relpath(rw_path, os.getcwd())}\n")

    # Re-load the saved file and re-classify: the committed npz (not the live
    # ensemble) is what the tests read, so confirm it round-trips.
    from kernel_viewer.io.loader import load_kernel
    rw2 = classify_any(load_kernel(rw_path))
    print(f"reloaded random walk -> {rw2.regime.value} / {rw2.coherence}")

    ok = (rw2.regime.value == "diffusive")
    print("\nCROSS-LIBRARY CHECK:", "OK" if ok else "UNEXPECTED -- regeneration drifted, stop and report")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
# (printed strings are ASCII-only so this runs on Windows cp932 consoles too)
