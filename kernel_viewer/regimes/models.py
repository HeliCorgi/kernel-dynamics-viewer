"""Synthetic kernel generators for testing and demos (deterministic)."""
import numpy as np
from ..core.kernel import Kernel


def diffusive(R=33, T=25, dt=0.25, D=0.6, weight=0.25):
    r = np.arange(R) - R // 2
    t = np.arange(T) * dt
    K = np.zeros((T, R))
    K[0, R // 2] = weight
    for i in range(1, T):
        var = 2 * D * t[i]
        p = np.exp(-r ** 2 / (2 * var))
        K[i] = weight * p / p.sum()
    return Kernel(r, t, K)


def ballistic(R=41, T=25, dt=0.25, v=2.0, width=1.0, weight=0.25):
    """Two counter-propagating wave packets: the textbook sound-peak kernel."""
    r = np.arange(R) - R // 2
    t = np.arange(T) * dt
    K = np.zeros((T, R))
    K[0, R // 2] = weight
    for i in range(1, T):
        p = np.exp(-(r - v * t[i]) ** 2 / (2 * width ** 2)) + np.exp(-(r + v * t[i]) ** 2 / (2 * width ** 2))
        K[i] = weight * p / p.sum()
    return Kernel(r, t, K)


def levy(R=61, T=25, dt=0.25, mu=1.0, c=0.7, weight=0.25):
    """Cauchy-type (alpha-stable, mu=1) spreading: heavy power-law tails."""
    r = np.arange(R) - R // 2
    t = np.arange(T) * dt
    K = np.zeros((T, R))
    K[0, R // 2] = weight
    for i in range(1, T):
        s = c * t[i] ** (1.0 / mu)
        p = s / (np.pi * (r ** 2 + s ** 2))
        K[i] = weight * p / p.sum()
    return Kernel(r, t, K)


def recurrent(R=33, T=49, dt=0.25, D=0.5, om=2.5, amp=0.6, weight=0.25):
    """Diffusive background plus an undamped breathing ring: integrable-like
    recurrent coherence."""
    r = np.arange(R) - R // 2
    t = np.arange(T) * dt
    K = np.zeros((T, R))
    K[0, R // 2] = weight
    for i in range(1, T):
        var = 2 * D * t[i]
        base = np.exp(-r ** 2 / (2 * var))
        ring = amp * (1 - np.cos(om * t[i])) * np.exp(-(np.abs(r) - 1.5) ** 2)
        p = base + ring
        K[i] = weight * p / p.sum()
    return Kernel(r, t, K)
