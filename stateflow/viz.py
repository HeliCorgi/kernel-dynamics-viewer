"""Publication-quality stateflow visualizations (matplotlib only).

Three required views:
  * axis-value timelines (continuous values + confidence band + transition marks);
  * 2D axis-space scatter (axis1 vs axis2, colored by regime read-out);
  * a distinct-information panel (axis 1 fixed while axis 2 varies).
Continuous axis values are always shown; never only the labels.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from .core import Trajectory


def plot_axis_timeline(traj: Trajectory, ax=None, color="#36c", label=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    v = traj.as_array()
    m = traj.mask
    ax.plot(traj.times[m], v[m], "o-", ms=3, color=color, label=label or traj.name)
    ax.plot(traj.times[~m], v[~m], "o", ms=3, color=color, alpha=0.2)
    if traj.confidence is not None:
        c = traj.confidence
        lo = v - (1 - np.clip(c, 0, 1)) * np.nanstd(v)
        hi = v + (1 - np.clip(c, 0, 1)) * np.nanstd(v)
        ax.fill_between(traj.times[m], lo[m], hi[m], color=color, alpha=0.12,
                        label="lower confidence = wider band")
    for tr in traj.transitions:
        ax.axvline(tr.time, color="#c44", ls="--", lw=1)
        ax.annotate(tr.kind, (tr.time, ax.get_ylim()[1]), fontsize=7,
                    color="#c44", rotation=90, va="top")
    ax.set_xlabel("t"); ax.set_ylabel(traj.name); ax.legend(fontsize=7.5)
    return ax


def plot_axis_space(axis1_vals, axis2_vals, regime_labels=None, ax=None,
                    xlabel="axis 1 (continuous)", ylabel="axis 2 (continuous)"):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))
    a1 = np.asarray(axis1_vals, dtype=float)
    a2 = np.asarray(axis2_vals, dtype=float)
    if regime_labels is None:
        ax.scatter(a1, a2, c="#36c", s=40)
    else:
        labs = list(dict.fromkeys(regime_labels))
        cmap = plt.get_cmap("tab10")
        for i, L in enumerate(labs):
            m = np.array([r == L for r in regime_labels])
            ax.scatter(a1[m], a2[m], color=cmap(i % 10), s=40, label=L)
        ax.legend(fontsize=7.5, title="regime read-out")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title("axis space (continuous; regime is a derived label)")
    return ax


def plot_distinct_information(fixed_vals, varying_vals, times=None, ax=None,
                              fixed_name="axis 1 (fixed)", varying_name="axis 2 (varies)"):
    """Show one axis ~constant while the other changes -- the visual proof that
    the second axis carries information the first does not."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    x = times if times is not None else np.arange(len(fixed_vals))
    ax.plot(x, fixed_vals, "s-", ms=3, color="#888", label=fixed_name)
    ax.plot(x, varying_vals, "o-", ms=3, color="#c44", label=varying_name)
    ax.set_xlabel("window"); ax.set_ylabel("continuous axis value")
    ax.set_title("distinct information: one axis fixed, the other varies")
    ax.legend(fontsize=7.5)
    return ax


def dashboard(axis1_traj: Trajectory, axis2_traj: Trajectory,
              axis_space=None, path=None):
    """Four-panel stateflow summary."""
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    plot_axis_timeline(axis1_traj, ax=ax[0, 0], color="#36c")
    plot_axis_timeline(axis2_traj, ax=ax[0, 1], color="#2a7")
    if axis_space is not None:
        a1, a2, labs = axis_space
        plot_axis_space(a1, a2, labs, ax=ax[1, 0])
    else:
        ax[1, 0].axis("off")
    # confidence evolution of both axes
    a = ax[1, 1]
    for tr, col, nm in ((axis1_traj, "#36c", axis1_traj.name),
                        (axis2_traj, "#2a7", axis2_traj.name)):
        if tr.confidence is not None:
            a.plot(tr.times, tr.confidence, "-", color=col, label=nm)
    a.set_xlabel("t"); a.set_ylabel("confidence"); a.set_ylim(-0.05, 1.05)
    a.set_title("per-axis confidence evolution"); a.legend(fontsize=7.5)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
        return path
    return fig
