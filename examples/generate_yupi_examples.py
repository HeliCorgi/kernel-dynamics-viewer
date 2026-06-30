"""Generate trajectory-ensemble example kernels with yupi (a cross-library check).

This script is NOT part of the package and is the ONLY place yupi is imported.
It synthesises two ensembles with yupi -- an independent implementation of 2D
Brownian motion and of Levy flights -- converts them into kernels via the
pure-NumPy helpers in ``kernel_viewer.io.trajectories``, classifies them with
the unmodified ``classify_any``, and saves the npz files bundled in examples/.

The scientific point: a correct classification of someone *else's* process is a
cross-library consistency check, not a tautology.

The crux (see README "Cross-library check (yupi)"): a square-lattice histogram
of a Levy ensemble CLIPS the heavy tail (rare long jumps fall outside any finite
box), which destroys the power-law signature. A bigger box does not fix it; the
fix is a radial DENSITY profile that bins the radial distance out to a high
quantile and divides by shell area, preserving the tail. So the random walk
uses ``ensemble_to_kernel2d`` (lattice) and the Levy ensemble uses
``ensemble_to_radial_kernel`` (radial).

Run:  pip install -e ".[examples]"  &&  python examples/generate_yupi_examples.py

Note: yupi's generator may not honour a global seed bit-for-bit, so the
classification a future run produces can differ slightly. The committed npz
files are the source of truth for the tests -- regeneration is a convenience /
API-drift check, not a reproducibility guarantee.
"""
import os

import numpy as np
from yupi.generators import RandomWalkGenerator

from kernel_viewer.io.trajectories import (ensemble_to_kernel2d,
                                           ensemble_to_radial_kernel)
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


def make_levy():
    """Independent 2D Levy flight -> radial-density 1D Kernel (heavy-tailed).

    Pareto(1.3)+1 step lengths give the heavy tail; the radial-density profile
    preserves it where a square lattice would clip it.
    """
    rng = np.random.default_rng(SEED)
    gen = RandomWalkGenerator(T=12.0, dim=2, dt=0.5, actions_prob=ACTIONS_PROB,
                              step_length_func=lambda s: rng.pareto(1.3, size=s) + 1.0,
                              seed=SEED)
    positions = _stack(gen.generate(15000))
    # pct=99.9 (not the helper's 99.5 default): the SAME tail-clipping crux that
    # rules out the lattice path also applies to the radial quantile. At pct=99.5
    # this ensemble's heaviest jumps are still clipped enough that the profile
    # reads as 'exponential' (positive log-log curvature). Keeping more of the
    # tail (99.9) exposes the genuine power law -> 'levy'. The process is a real
    # Levy flight (step magnitudes are heavy-tailed: median ~3, 99.9pct ~330);
    # this is a quantile choice that preserves the tail, not a workaround.
    return ensemble_to_radial_kernel(positions, pct=99.9)


def main():
    rw = make_random_walk()
    rw_path = os.path.join(HERE, "yupi_randomwalk2d.npz")
    np.savez_compressed(rw_path, x=rw.x, y=rw.y, t=rw.t, K=rw.K.astype(np.float32))
    rep_rw = classify_any(rw)
    print("yupi random walk (lattice Kernel2D):")
    print(rep_rw.summary())
    print(f"-> {os.path.relpath(rw_path, os.getcwd())}\n")

    lv = make_levy()
    lv_path = os.path.join(HERE, "yupi_levy2d_radial.npz")
    np.savez_compressed(lv_path, r=lv.r, t=lv.t, K=lv.K.astype(np.float32))
    rep_lv = classify_any(lv)
    print("yupi Levy flight (radial-density 1D Kernel):")
    print(rep_lv.summary())
    print(f"-> {os.path.relpath(lv_path, os.getcwd())}\n")

    # Re-load the saved files and re-classify: the committed npz (not the live
    # ensemble) is what the tests read, so confirm it round-trips to the same
    # regime before we rely on it.
    from kernel_viewer.io.loader import load_kernel
    rw2 = classify_any(load_kernel(rw_path))
    lv2 = classify_any(load_kernel(lv_path))
    print(f"reloaded random walk -> {rw2.regime.value} / {rw2.coherence}")
    print(f"reloaded Levy        -> {lv2.regime.value} / tail={lv2.features.get('tail shape')}")

    ok = (rw2.regime.value == "diffusive" and lv2.regime.value == "levy")
    print("\nCROSS-LIBRARY CHECK:", "OK" if ok else "UNEXPECTED -- see handoff section 8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
# (printed strings are ASCII-only so this runs on Windows cp932 consoles too)
