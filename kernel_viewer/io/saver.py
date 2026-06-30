import json
import numpy as np


def save_report(report, path):
    obj = {
        "regime": report.regime.value,
        "coherence": report.coherence,
        "exponent": None if not np.isfinite(report.exponent) else round(float(report.exponent), 4),
        "features": report.features,
        "notes": report.notes,
    }
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path


def save_spectrum(spectrum, path):
    """Save a Spectrum to .npz: k, omega, S_kw, transform_type, warnings."""
    np.savez(path, k=spectrum.k, omega=spectrum.omega, S_kw=spectrum.S_kw,
             transform_type=getattr(spectrum, "transform_type", "unknown"),
             warnings=np.array(getattr(spectrum, "warnings", []), dtype=object))
    return path
