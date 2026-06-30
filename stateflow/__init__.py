"""stateflow: a trajectory library for multi-axis regime analysis.

The center of this library is the *evolution of measurements*, not the regime.
A small number of independently computed continuous axes are tracked as
trajectories; transitions are derived from those trajectories; a regime label
is a terminal, optional read-out of a location in axis space.

Two new abstractions only: ``FeatureSeries`` (container) and ``Trajectory``
(one class, used at three levels: feature / axis / transition). ``Axis`` is a
protocol, not a stateful object. Everything else is reused from existing code.
"""
from .core import FeatureSeries, Trajectory, Transition, Axis, TransitionAnalyzer
from .kernel_plugin import KernelExtractor, TransportAxis, CoherenceAxis

__version__ = "0.1.0"
__all__ = [
    "FeatureSeries", "Trajectory", "Transition", "Axis", "TransitionAnalyzer",
    "KernelExtractor", "TransportAxis", "CoherenceAxis",
]
