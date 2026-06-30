"""stateflow-eeg: a synthetic external extractor proving the same core handles
a second domain. No real EEG, no new dependencies -- a deterministic generator
plus two axes computed by Welch-style periodograms and a band phase/amplitude
coupling estimate (SciPy only).

This is a PLUGIN: the core never learns these meanings. The axes return
continuous values (spectral slope; coupling-strength trajectory); labels are
optional and downstream, exactly as in the kernel plugin.
"""
from __future__ import annotations
import numpy as np
from scipy import signal

from ..core import FeatureSeries, Trajectory, Transition


def synth_eeg(fs_hz: float = 256.0, dur_s: float = 20.0, seed: int = 0,
              couple_onset_s: float = 10.0) -> FeatureSeries:
    """Generate a 1/f (pink-ish) EEG-like signal whose theta-gamma coupling
    switches on at ``couple_onset_s``. Returns a FeatureSeries holding the raw
    signal (one feature column) -- the axes do the spectral work."""
    rng = np.random.default_rng(seed)
    n = int(fs_hz * dur_s)
    t = np.arange(n) / fs_hz
    # proper 1/f^alpha noise by shaping white noise in the frequency domain
    alpha = 1.7  # pink-ish target slope
    Xf = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    shape = np.ones_like(f); shape[1:] = 1.0 / np.power(f[1:], alpha / 2.0)
    pink = np.fft.irfft(Xf * shape, n=n)
    pink = pink / np.std(pink)
    theta = np.sin(2 * np.pi * 6 * t)
    gamma = np.sin(2 * np.pi * 40 * t)
    # theta-gamma PAC after onset: gamma amplitude locked to theta phase
    pac_gate = np.clip((t - couple_onset_s) / 1.0, 0.0, 1.0)
    amp_env = 1.0 + pac_gate * 1.5 * (1 + theta) / 2.0
    x = 1.2 * pink + 0.8 * theta + 0.9 * amp_env * gamma
    return FeatureSeries(
        time=t, features=x, feature_names=["eeg"],
        metadata={"provenance": "stateflow.extractors.eeg.synth_eeg",
                  "method": "synthetic 1/f + theta + (PAC-gated) gamma",
                  "sampling_hz": fs_hz, "couple_onset_s": couple_onset_s,
                  "units": "arbitrary"},
    )


class SpectralSlopeAxis:
    """Axis 1 -- spectral slope. Continuous descriptor: the 1/f exponent from a
    log-log fit of the Welch PSD over sliding windows. Confidence: R^2 of the
    fit. discretize() is optional (sparse/natural/pink/white)."""

    name = "spectral_slope"

    def __init__(self, window_s: float = 2.0, step_s: float = 1.0,
                 band=(2.0, 50.0)) -> None:
        self.window_s = window_s
        self.step_s = step_s
        self.band = band

    def _windows(self, fs: FeatureSeries):
        sr = fs.metadata["sampling_hz"]
        w = int(self.window_s * sr); st = int(self.step_s * sr)
        x = fs.column("eeg")
        for s in range(0, len(x) - w + 1, st):
            yield fs.time[s + w // 2], x[s:s + w], sr

    def compute(self, fs: FeatureSeries) -> Trajectory:
        times, slopes, r2s = [], [], []
        for tc, seg, sr in self._windows(fs):
            f, p = signal.welch(seg, fs=sr, nperseg=min(len(seg), 256))
            m = (f >= self.band[0]) & (f <= self.band[1]) & (p > 0)
            if m.sum() < 4:
                times.append(tc); slopes.append(np.nan); r2s.append(0.0); continue
            lf, lp = np.log(f[m]), np.log(p[m])
            A = np.vstack([lf, np.ones_like(lf)]).T
            coef, *_ = np.linalg.lstsq(A, lp, rcond=None)
            pred = A @ coef
            ss = ((lp - lp.mean()) ** 2).sum()
            r2 = 1 - ((lp - pred) ** 2).sum() / ss if ss > 0 else 0.0
            times.append(tc); slopes.append(-coef[0]); r2s.append(float(r2))
        traj = Trajectory(times=np.array(times), values=np.array(slopes),
                          confidence=np.array(r2s), mask=np.isfinite(slopes),
                          name="spectral_slope_exponent")
        return traj

    def confidence(self, fs: FeatureSeries) -> np.ndarray:
        return self.compute(fs).confidence

    def discretize(self, traj: Trajectory) -> list[str]:
        out = []
        for v in traj.as_array():
            if not np.isfinite(v):
                out.append("unknown")
            elif v < 0.5:
                out.append("white")
            elif v < 1.5:
                out.append("natural")
            elif v < 2.5:
                out.append("pink")
            else:
                out.append("sparse")
        return out


class CouplingAxis:
    """Axis 2 -- cross-frequency coupling. NATIVE continuous descriptor: a
    theta-gamma phase-amplitude coupling-strength TRAJECTORY (modulation index
    over sliding windows). Confidence: the coupling magnitude itself."""

    name = "coupling"

    def __init__(self, window_s: float = 2.0, step_s: float = 1.0,
                 theta=(4.0, 8.0), gamma=(30.0, 50.0)) -> None:
        self.window_s = window_s; self.step_s = step_s
        self.theta = theta; self.gamma = gamma

    def compute(self, fs: FeatureSeries) -> Trajectory:
        sr = fs.metadata["sampling_hz"]
        x = fs.column("eeg")
        w = int(self.window_s * sr); st = int(self.step_s * sr)
        bt = signal.butter(3, np.array(self.theta) / (sr / 2), btype="band")
        bg = signal.butter(3, np.array(self.gamma) / (sr / 2), btype="band")
        times, mis = [], []
        for s in range(0, len(x) - w + 1, st):
            seg = x[s:s + w]
            ph = np.angle(signal.hilbert(signal.filtfilt(*bt, seg)))
            am = np.abs(signal.hilbert(signal.filtfilt(*bg, seg)))
            # modulation index: mean amplitude in phase bins, normalized spread
            nb = 18
            bins = np.linspace(-np.pi, np.pi, nb + 1)
            idx = np.digitize(ph, bins) - 1
            mvec = np.array([am[idx == b].mean() if np.any(idx == b) else 0.0 for b in range(nb)])
            if mvec.sum() == 0 or mvec.max() == 0:
                mi = 0.0
            else:
                # modulation depth of the phase-amplitude histogram: 0 = flat
                # (no coupling), -> 1 = amplitude fully concentrated at one
                # theta phase. More interpretable and higher dynamic range
                # than the normalized-KL modulation index for short windows.
                mi = float((mvec.max() - mvec.min()) / (mvec.max() + mvec.min()))
            times.append(fs.time[s + w // 2]); mis.append(mi)
        mis = np.array(mis)
        traj = Trajectory(times=np.array(times), values=mis,
                          confidence=mis, mask=np.isfinite(mis),
                          name="theta_gamma_modulation_index")
        return traj

    def confidence(self, fs: FeatureSeries) -> np.ndarray:
        return self.compute(fs).as_array()

    def discretize(self, traj: Trajectory) -> list[str]:
        out = []
        for v in traj.as_array():
            if not np.isfinite(v):
                out.append("unknown")
            elif v < 0.25:
                out.append("decoupled")
            elif v < 0.55:
                out.append("loosely_coupled")
            else:
                out.append("tightly_coupled")
        return out
