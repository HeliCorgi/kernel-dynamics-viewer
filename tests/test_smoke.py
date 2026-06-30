import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kernel_viewer.regimes import models
from kernel_viewer.regimes.classifier import classify, Regime


def test_ground_truth():
    expect = {
        "diffusive": (Regime.DIFFUSIVE, "MONOTONE"),
        "ballistic": (Regime.BALLISTIC, "RING_PERSISTENT"),
        "levy": (Regime.LEVY, "MONOTONE"),
        "recurrent": (Regime.DIFFUSIVE, "RING_RECURRENT"),
    }
    for name, (er, ec) in expect.items():
        rep = classify(getattr(models, name)())
        assert rep.regime == er, (name, rep.regime)
        assert rep.coherence == ec, (name, rep.coherence)


def test_real_examples():
    from kernel_viewer.io.loader import load_kernel
    base = os.path.join(os.path.dirname(__file__), "..", "examples")
    rep = classify(load_kernel(os.path.join(base, "xxz_Sx_integrable.npz")))
    assert rep.coherence == "RING_RECURRENT"
    rep = classify(load_kernel(os.path.join(base, "ising_q_chaotic.npz")))
    assert rep.coherence == "RING_TRANSIENT"


def test_ground_truth_2d():
    from kernel_viewer.regimes import models2d
    from kernel_viewer.regimes.classifier import classify2d
    expect = {
        "diffusive2d": (Regime.DIFFUSIVE, "MONOTONE"),
        "ballistic2d": (Regime.BALLISTIC, "RING_PERSISTENT"),
        "levy2d": (Regime.LEVY, "MONOTONE"),
        "randomwalk2d": (Regime.DIFFUSIVE, "MONOTONE"),
    }
    for name, (er, ec) in expect.items():
        rep = classify2d(getattr(models2d, name)())
        assert rep.regime == er, (name, rep.regime)
        assert rep.coherence == ec, (name, rep.coherence)
    rep = classify2d(models2d.anisotropic2d())
    assert any("anisotropic" in n for n in rep.notes), "C4 anisotropy warning missing"


def test_loader_dispatch():
    import numpy as np, tempfile
    from kernel_viewer.io.loader import load_kernel
    from kernel_viewer.core.kernel2d import Kernel2D
    from kernel_viewer.regimes import models2d
    kn = models2d.diffusive2d(L=21, T=5)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        np.savez(f.name, x=kn.x, y=kn.y, t=kn.t, K=kn.K)
        assert isinstance(load_kernel(f.name), Kernel2D)


def test_ground_truth_3d():
    from kernel_viewer.regimes import models3d
    from kernel_viewer.regimes.classifier import classify3d
    expect = {
        "diffusive3d": (Regime.DIFFUSIVE, "MONOTONE"),
        "ballistic3d": (Regime.BALLISTIC, "RING_PERSISTENT"),
        "levy3d": (Regime.LEVY, "MONOTONE"),
        "randomwalk3d": (Regime.DIFFUSIVE, "MONOTONE"),
    }
    for name, (er, ec) in expect.items():
        rep = classify3d(getattr(models3d, name)())
        assert rep.regime == er, (name, rep.regime)
        assert rep.coherence == ec, (name, rep.coherence)
        assert rep.features["dimensionality"] == 3
    rep = classify3d(models3d.cubic3d())
    assert any("anisotropic" in n for n in rep.notes), "cubic (Q4) anisotropy warning missing"
    assert rep.features["isotropy_score"] < 0.9


def test_loader_3d():
    import numpy as np, tempfile
    from kernel_viewer.io.loader import load_kernel
    from kernel_viewer.core.kernel3d import Kernel3D
    from kernel_viewer.regimes import models3d
    kn = models3d.diffusive3d(L=17, T=4)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        np.savez(f.name, x=kn.x, y=kn.y, z=kn.z, t=kn.t, K=kn.K.astype(np.float32))
        k2 = load_kernel(f.name)
        assert isinstance(k2, Kernel3D)


def test_radial_spectrum_gaussian_symbol():
    """Hankel (2D) and spherical-Bessel (3D) spatial transforms must match the
    exact Gaussian symbol C(k,t) = W exp(-D k^2 t) to ~10% (integer-bin
    discretization)."""
    import numpy as np
    from kernel_viewer.regimes.models2d import diffusive2d
    from kernel_viewer.regimes import models3d
    for kn, D in ((diffusive2d(D=0.6), 0.6), (models3d.diffusive3d(D=0.6), 0.6)):
        rk = kn.radial()
        rho, P = rk.symmetrized()
        counts = rk.bin_counts
        dim = rk.geometry_dim
        from scipy.special import j0
        basis = (lambda x: j0(x)) if dim == 2 else (lambda x: np.sinc(x / np.pi))
        k = 0.8
        Ck = np.array([(np.nan_to_num(P[i]) * counts * basis(k * rho)).sum()
                       for i in range(len(kn.t))])
        pred = 0.25 * np.exp(-D * k * k * kn.t)
        ratio = Ck[1:6] / pred[1:6]
        assert np.all(np.abs(ratio - 1) < 0.12), (dim, ratio)


def test_structure_factor_api():
    """v0.4: unified API, transform dispatch, anisotropy warning."""
    from kernel_viewer.metrics.fourier import compute_structure_factor
    from kernel_viewer.regimes import models, models2d, models3d
    sp = compute_structure_factor(models.diffusive())
    assert sp.transform_type == "cosine"
    sp = compute_structure_factor(models2d.diffusive2d())
    assert sp.transform_type == "hankel_j0" and not sp.warnings
    sp = compute_structure_factor(models3d.diffusive3d())
    assert sp.transform_type == "spherical_bessel_j0"
    sp = compute_structure_factor(models3d.cubic3d())
    assert sp.warnings, "anisotropic kernel must attach a warning"


def test_dispersion_recovery():
    """v0.4 physics check: omega_peak(k) recovers the front dispersion.
    2D shell: J0(kvt) ~ t^{-1/2} oscillation -> integrable peak at omega=vk;
    slope must be within ~15% of v. 3D shell: j0(kvt) ~ 1/t envelope makes
    S(k, omega) a PLATEAU with an edge at omega=vk (textbook surface
    dilution), so omega_peak is linear in k but systematically BELOW v --
    assert linearity and the documented bias direction only."""
    import numpy as np
    from kernel_viewer.metrics.fourier import compute_structure_factor
    from kernel_viewer.regimes.models2d import ballistic2d, diffusive2d
    from kernel_viewer.regimes import models3d

    sp = compute_structure_factor(ballistic2d())  # v = 2.5
    k, om = sp.k, sp.peak_frequency()
    m = (k >= 0.15 * np.pi) & (k <= 0.55 * np.pi) & (om > 0)
    slope = float((k[m] @ om[m]) / (k[m] @ k[m]))
    corr = float(np.corrcoef(k[m], om[m])[0, 1])
    assert corr > 0.99 and 2.0 < slope < 2.75, (slope, corr)

    sp = compute_structure_factor(models3d.ballistic3d())  # v = 3.0
    k, om = sp.k, sp.peak_frequency()
    m = (k >= 0.15 * np.pi) & (k <= 0.55 * np.pi) & (om > 0)
    slope = float((k[m] @ om[m]) / (k[m] @ k[m]))
    corr = float(np.corrcoef(k[m], om[m])[0, 1])
    assert corr > 0.98 and 1.4 < slope < 3.1, (slope, corr)

    spd = compute_structure_factor(diffusive2d())
    dom = spd.omega[1] - spd.omega[0]
    assert float(spd.peak_frequency().max()) <= dom + 1e-9


def test_ground_truth_4d():
    from kernel_viewer.regimes import models4d
    from kernel_viewer.regimes.classifier import classify4d
    expect = {
        "diffusive4d": (Regime.DIFFUSIVE, "MONOTONE"),
        "levy4d": (Regime.LEVY, "MONOTONE"),
        "randomwalk4d": (Regime.DIFFUSIVE, "MONOTONE"),
    }
    for name, (er, ec) in expect.items():
        rep = classify4d(getattr(models4d, name)())
        assert rep.regime == er, (name, rep.regime)
        assert rep.coherence == ec, (name, rep.coherence)
        assert rep.features["dimensionality"] == 4
    # Documented 4D box-physics (README "wavefront dilution vs dimension"):
    # in feasible boxes the inscribed hypersphere clips the shell's outer
    # tail and the width transient dominates the short clean window, so a
    # true ballistic shell reads superdiffusive (p ~ 1.7) with a persistent
    # front. Assert that documented behavior, not an unreachable ideal.
    rep = classify4d(models4d.ballistic4d())
    assert rep.regime in (Regime.BALLISTIC, Regime.SUPERDIFFUSIVE), rep.regime
    assert rep.coherence == "RING_PERSISTENT" and rep.exponent > 1.6, (rep.coherence, rep.exponent)
    rep = classify4d(models4d.cubic4d())
    assert any("anisotropic" in n for n in rep.notes)
    assert rep.features["isotropy_score"] < 0.9


def test_loader_4d_and_symbol():
    import numpy as np, tempfile
    from kernel_viewer.io.loader import load_kernel
    from kernel_viewer.core.kernel4d import Kernel4D
    from kernel_viewer.regimes import models4d
    from kernel_viewer.metrics.fourier import compute_structure_factor
    from scipy.special import j1
    kn = models4d.diffusive4d(L=13, T=4)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        np.savez(f.name, x=kn.x, y=kn.y, z=kn.z, w=kn.w, t=kn.t, K=kn.K.astype(np.float32))
        assert isinstance(load_kernel(f.name), Kernel4D)
    # exact Gaussian symbol in d=4 with the 2 J1(x)/x zonal basis
    kn = models4d.diffusive4d(D=0.5)
    rk = kn.radial(); rho, P = rk.symmetrized(); counts = rk.bin_counts
    k = 0.8
    b = np.ones_like(rho); m = rho > 0
    b[m] = 2 * j1(k * rho[m]) / (k * rho[m])
    Ck = np.array([(np.nan_to_num(P[i]) * counts * b).sum() for i in range(len(kn.t))])
    pred = 0.25 * np.exp(-0.5 * k * k * kn.t)
    assert np.all(np.abs(Ck[1:5] / pred[1:5] - 1) < 0.12)
    sp = compute_structure_factor(kn)
    assert sp.transform_type == "generalized_hankel_j1" and not sp.warnings
    sp = compute_structure_factor(models4d.cubic4d())
    assert sp.warnings, "4D anisotropy warning missing"


def test_track_transition():
    """v0.6: sliding-window tracking flags a broadband->coherent onset."""
    from kernel_viewer.regimes.models2d import coherent_onset2d
    from kernel_viewer.metrics.tracking import track_regimes
    recs = track_regimes(coherent_onset2d(), window=28, stride=6)
    assert recs[0]["coherence"] == "MONOTONE"
    assert recs[-1]["coherence"] == "RING_RECURRENT"
    flags = [i for i, r in enumerate(recs) if r["transition"]]
    assert flags and recs[flags[0]]["t_end"] >= 7.5, "transition must be flagged once the window reaches onset"


def test_load_any_mat_and_csv():
    """v0.8: load_any reads .mat and .csv and yields identical regime results
    to the .npz path, proving the loaders don't perturb the analysis."""
    import tempfile, warnings
    import numpy as np
    from scipy.io import savemat
    from kernel_viewer.io.loader import load_kernel, load_any
    from kernel_viewer.regimes.classifier import classify_any

    base = os.path.join(os.path.dirname(__file__), "..", "examples")
    npz = os.path.join(base, "xxz_Sx_integrable.npz")
    kn = load_kernel(npz)
    ref = classify_any(kn)
    T, R = kn.K.shape
    tmp = tempfile.mkdtemp()

    # load_any on the npz must be byte-identical to load_kernel's result.
    rep_npz = classify_any(load_any(npz))
    assert rep_npz.regime == ref.regime and rep_npz.coherence == ref.coherence

    # --- .mat round trip (time already on axis 0) ---
    mat = os.path.join(tmp, "k.mat")
    savemat(mat, {"K": kn.K, "t": kn.t, "r": kn.r})
    km = load_any(mat)
    rm = classify_any(km)
    assert km.K.shape == (T, R)
    assert rm.regime == ref.regime and rm.coherence == ref.coherence

    # --- .mat transpose detection: store as (R, T) and confirm correction ---
    matT = os.path.join(tmp, "kT.mat")
    savemat(matT, {"K": kn.K.T, "t": kn.t, "r": kn.r})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kt = load_any(matT)
    assert kt.K.shape == (T, R), "(R, T) mat must be transposed to (T, R)"
    assert np.allclose(kt.K, kn.K)
    assert any("transpos" in str(w.message).lower() for w in caught)
    rt = classify_any(kt)
    assert rt.regime == ref.regime and rt.coherence == ref.coherence

    # --- .csv long format (t, r, K) round trip ---
    long_csv = os.path.join(tmp, "k_long.csv")
    rows = [[kn.t[i], kn.r[j], kn.K[i, j]] for i in range(T) for j in range(R)]
    np.savetxt(long_csv, np.array(rows), delimiter=",")
    kl = load_any(long_csv)
    rl = classify_any(kl)
    assert kl.K.shape == (T, R)
    assert rl.regime == ref.regime and rl.coherence == ref.coherence

    # --- .csv labeled matrix (NaN corner, r header, t column) round trip ---
    mat_csv = os.path.join(tmp, "k_matrix.csv")
    M = np.full((T + 1, R + 1), np.nan)
    M[0, 1:] = kn.r
    M[1:, 0] = kn.t
    M[1:, 1:] = kn.K
    np.savetxt(mat_csv, M, delimiter=",")
    kc = load_any(mat_csv)
    rc = classify_any(kc)
    assert kc.K.shape == (T, R)
    assert np.allclose(kc.r, kn.r) and np.allclose(kc.t, kn.t)
    assert rc.regime == ref.regime and rc.coherence == ref.coherence

    # --- CSV cannot carry a 2D+ kernel: must raise NotImplementedError ---
    csv2d = os.path.join(tmp, "k2d.csv")
    np.savetxt(csv2d, np.zeros((12, 4)), delimiter=",")
    raised = False
    try:
        load_any(csv2d, layout="long")
    except NotImplementedError:
        raised = True
    assert raised, "a 4-column long CSV (2D) must raise NotImplementedError"


def test_trajectory_helpers():
    """v0.9: pure-NumPy unit tests of the trajectory->kernel helpers (NO yupi).
    A light-tailed (Gaussian) ensemble must diffuse (sigma2 ~ t through the
    lattice path); a heavy-tailed (Cauchy) ensemble's radial-density profile
    must expose a power-law tail. This proves the helpers are correct
    independently of yupi."""
    import numpy as np
    from kernel_viewer.io.trajectories import (ensemble_to_kernel2d,
                                               ensemble_to_radial_kernel)
    from kernel_viewer.regimes.classifier import classify_any

    N, T = 4000, 12
    # --- light-tailed Gaussian random walk -> sigma2 increases ~linearly ---
    rng = np.random.default_rng(0)
    steps = rng.normal(0.0, 1.0, size=(N, T - 1, 2))
    pos = np.zeros((N, T, 2)); pos[:, 1:, :] = np.cumsum(steps, axis=1)
    s2 = ensemble_to_kernel2d(pos, L=41).sigma2()
    s2 = s2[np.isfinite(s2)]
    assert np.all(np.diff(s2) > 0), "Gaussian ensemble sigma2 must increase in time"
    tt = np.arange(len(s2), dtype=float)
    assert np.corrcoef(tt, s2)[0, 1] > 0.97, "sigma2 must grow ~linearly (diffusive)"

    # --- heavy-tailed Cauchy walk -> radial profile reads as power-law ---
    rng2 = np.random.default_rng(1)
    csteps = rng2.standard_cauchy(size=(N, T - 1, 2))
    cpos = np.zeros((N, T, 2)); cpos[:, 1:, :] = np.cumsum(csteps, axis=1)
    rep = classify_any(ensemble_to_radial_kernel(cpos, pct=99.5))
    assert rep.features.get("tail shape", "").startswith("powerlaw"), \
        f"heavy-tailed radial profile must be powerlaw, got {rep.features.get('tail shape')}"


def test_yupi_examples_classify():
    """v0.9.1: the committed yupi random-walk cross-library example classifies
    as expected through the normal load path. Reads the committed npz, so needs
    no yupi -- the core job stays yupi-free. (The Lévy example was removed in
    v0.9.1: its radial profile is too large to classify quickly and the
    tail-shape heuristic is not robust for heavy tails -- out of scope.)"""
    from kernel_viewer.io.loader import load_kernel
    from kernel_viewer.regimes.classifier import classify_any, Regime
    base = os.path.join(os.path.dirname(__file__), "..", "examples")
    rw = classify_any(load_kernel(os.path.join(base, "yupi_randomwalk2d.npz")))
    assert rw.regime == Regime.DIFFUSIVE, ("randomwalk", rw.regime)
    assert rw.coherence == "MONOTONE", ("randomwalk", rw.coherence)


if __name__ == "__main__":
    test_ground_truth(); test_real_examples(); test_ground_truth_2d(); test_loader_dispatch()
    test_ground_truth_3d(); test_loader_3d(); test_radial_spectrum_gaussian_symbol()
    test_structure_factor_api(); test_dispersion_recovery()
    test_ground_truth_4d(); test_loader_4d_and_symbol()
    test_track_transition()
    test_load_any_mat_and_csv()
    test_trajectory_helpers()
    test_yupi_examples_classify()
    print("ALL TESTS PASS")
