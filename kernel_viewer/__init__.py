"""Kernel Dynamics Viewer: regime diagnostics for spatiotemporal kernels K(r,t)."""
from .core.kernel import Kernel
from .core.spectral import Spectrum, compute_spectrum
from .io.loader import load_kernel, load_any
from .regimes.classifier import classify, classify_any, RegimeReport
__version__ = "1.0.0"
__all__ = ["Kernel", "Spectrum", "compute_spectrum", "load_kernel", "load_any",
           "classify", "classify_any", "RegimeReport"]
