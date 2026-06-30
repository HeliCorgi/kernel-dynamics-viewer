"""stateflow core: the two new abstractions plus the transition analyzer.

Design constraints honored here:
  * Axes return CONTINUOUS values; labels are optional and downstream.
  * Exactly two new abstractions: FeatureSeries (container) and Trajectory
    (one class, three uses). Axis is a typing.Protocol (no state, no new
    concrete base class). Transition is a tiny record.
  * No HMM / clustering / NN / optimizer / sklearn. NumPy only in core.
  * Masks propagate to every downstream operation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable, Any, Optional
import numpy as np


@dataclass
class FeatureSeries:
    """A time-indexed container of measured feature vectors. Domain-agnostic:
    it knows nothing about what the features mean.

    Parameters
    ----------
    time : (T,) float array of sample times (or window centers).
    features : (T,) or (T, F) float array of measured quantities.
    feature_names : names for the F columns (or the single feature).
    metadata : provenance dict -- source, method, sampling_interval, units,
        preprocessing, known artifacts. Recorded for reproducibility.
    uncertainty : optional, broadcastable to ``features``.
    mask : (T,) bool, True = usable. Defaults to all-True. Every downstream
        operation (axes, trajectories, transitions, plots) respects it.
    """
    time: np.ndarray
    features: np.ndarray
    feature_names: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    uncertainty: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float)
        self.features = np.asarray(self.features, dtype=float)
        if self.features.ndim == 1:
            if len(self.features) != len(self.time):
                raise ValueError("features (T,) length must match time")
        elif self.features.ndim == 2:
            if self.features.shape[0] != len(self.time):
                raise ValueError("features (T, F) first axis must match time")
        else:
            raise ValueError("features must be (T,) or (T, F)")
        if not self.feature_names:
            f = 1 if self.features.ndim == 1 else self.features.shape[1]
            self.feature_names = [f"f{i}" for i in range(f)]
        if self.mask is None:
            self.mask = np.ones(len(self.time), dtype=bool)
        else:
            self.mask = np.asarray(self.mask, dtype=bool)
            if self.mask.shape != (len(self.time),):
                raise ValueError("mask must be (T,) bool")
        self.metadata.setdefault("provenance", "unspecified")

    @property
    def n_features(self) -> int:
        return 1 if self.features.ndim == 1 else self.features.shape[1]

    def column(self, name_or_index) -> np.ndarray:
        """Return one feature column by name or index (1D array)."""
        if isinstance(name_or_index, str):
            j = self.feature_names.index(name_or_index)
        else:
            j = int(name_or_index)
        return self.features if self.features.ndim == 1 else self.features[:, j]


@dataclass
class Transition:
    """A point where a trajectory changes character. ``kind`` is a free string
    (e.g. a coherence-class change "MONOTONE->RING_RECURRENT"); ``index`` is
    the position in the trajectory; ``time`` the corresponding time."""
    index: int
    time: float
    kind: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """The evolution object -- ONE class used at three levels (feature / axis /
    transition trajectory). A sequence of measurements over time or sliding
    windows, plus optional per-step confidence, mask, and transition structure.

    ``values`` is continuous: either a (N,) / (N, K) float array (scalar- or
    vector-per-step) or a length-N list of richer per-step objects (e.g. the
    episode history of a ring trajectory). The continuous value is primary;
    discretization is never stored here.
    """
    times: np.ndarray
    values: Any
    confidence: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    transitions: list[Transition] = field(default_factory=list)
    name: str = ""

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=float)
        n = len(self.times)
        if isinstance(self.values, np.ndarray):
            if self.values.shape[0] != n:
                raise ValueError("values first axis must match times")
        elif isinstance(self.values, list):
            if len(self.values) != n:
                raise ValueError("values list length must match times")
        if self.mask is None:
            self.mask = np.ones(n, dtype=bool)
        else:
            self.mask = np.asarray(self.mask, dtype=bool)
        if self.confidence is not None:
            self.confidence = np.asarray(self.confidence, dtype=float)

    def as_array(self) -> np.ndarray:
        """Continuous values as a float array (raises if values are objects)."""
        if not isinstance(self.values, np.ndarray):
            raise TypeError("trajectory values are not a numeric array")
        return self.values


@runtime_checkable
class Axis(Protocol):
    """An independent measurement mapping a FeatureSeries to a CONTINUOUS
    descriptor trajectory. A Protocol, not a base class: any object with these
    three methods is an Axis, so plugins need not subclass anything.

    Contract:
      * ``compute`` returns continuous values (a Trajectory), never labels.
      * ``discretize`` is OPTIONAL and must never be required to obtain or
        interpret the axis; it is lossless to the underlying values.
      * Axes are computed independently; no data flows between them.
    """
    name: str

    def compute(self, fs: FeatureSeries) -> Trajectory: ...

    def confidence(self, fs: FeatureSeries) -> np.ndarray: ...

    def discretize(self, traj: Trajectory) -> list[str]: ...


class TransitionAnalyzer:
    """Derive a TransitionTrajectory from any AxisTrajectory, generalizing
    ``kernel_viewer.metrics.tracking``. Two modes:

      * categorical: a per-step label sequence (e.g. axis.discretize output) ->
        a transition wherever the label changes (trailing masked/None steps do
        not emit a transition, matching tracking.py semantics);
      * threshold: a per-step scalar crossing ``threshold`` -> a transition at
        each upward/downward crossing.

    Also computes dwell times, persistence (longest dwell), and recurrence
    (number of distinct episodes of a given label/above-threshold state).
    """

    def __init__(self, min_dwell: int = 1) -> None:
        self.min_dwell = int(min_dwell)

    def from_labels(self, times: np.ndarray, labels: list[Optional[str]],
                    mask: Optional[np.ndarray] = None) -> Trajectory:
        times = np.asarray(times, dtype=float)
        if mask is None:
            mask = np.array([lab is not None for lab in labels])
        transitions: list[Transition] = []
        prev = None
        for i, lab in enumerate(labels):
            if lab is None or not mask[i]:
                continue
            if prev is not None and lab != prev:
                transitions.append(Transition(index=i, time=float(times[i]),
                                              kind=f"{prev}->{lab}"))
            prev = lab
        traj = Trajectory(times=times, values=np.arange(len(times), dtype=float),
                          mask=np.asarray(mask, dtype=bool),
                          transitions=transitions, name="transitions")
        traj.labels = labels  # attached, not part of the generic schema
        return traj

    def from_threshold(self, times: np.ndarray, scalar: np.ndarray,
                       threshold: float) -> Trajectory:
        times = np.asarray(times, dtype=float)
        scalar = np.asarray(scalar, dtype=float)
        above = scalar > threshold
        transitions: list[Transition] = []
        for i in range(1, len(scalar)):
            if np.isfinite(scalar[i]) and np.isfinite(scalar[i - 1]) and above[i] != above[i - 1]:
                kind = "rise" if above[i] else "fall"
                transitions.append(Transition(index=i, time=float(times[i]), kind=kind))
        traj = Trajectory(times=times, values=scalar, mask=np.isfinite(scalar),
                          transitions=transitions, name="threshold_transitions")
        traj.above = above
        return traj

    @staticmethod
    def dwell_times(traj: Trajectory) -> list[float]:
        """Time spans between consecutive transitions (plus the head and tail
        segments)."""
        if len(traj.times) == 0:
            return []
        edges = [traj.times[0]] + [tr.time for tr in traj.transitions] + [traj.times[-1]]
        return [float(b - a) for a, b in zip(edges[:-1], edges[1:])]

    @staticmethod
    def persistence(traj: Trajectory) -> float:
        d = TransitionAnalyzer.dwell_times(traj)
        return float(max(d)) if d else 0.0

    @staticmethod
    def recurrence(traj: Trajectory) -> int:
        """Number of distinct episodes = transitions into a state. For labels,
        counts label changes + 1 (number of constant runs); for threshold,
        counts upward crossings."""
        if hasattr(traj, "above"):
            return int(np.sum([tr.kind == "rise" for tr in traj.transitions]))
        return len(traj.transitions) + 1 if len(traj.times) else 0
