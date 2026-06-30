"""Headless smoke test for the Streamlit GUI (v0.8).

Runs ONLY in the dedicated `gui` CI job (which installs the `.[gui]` extra).
It imports the app module's analysis wiring and runs one full analysis
end-to-end on bundled examples WITHOUT launching a browser or a Streamlit
server. main() (the UI) is never called.

If Streamlit is not installed (e.g. the core CI job, or a minimal local
install) the test skips cleanly so it can never break the Streamlit-free core.
"""
import importlib.util
import os
import sys
import tempfile

import numpy as np
from matplotlib.figure import Figure

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)


def _load_app_module():
    """Import app/streamlit_app.py without executing its `main()` UI."""
    path = os.path.join(ROOT, "app", "streamlit_app.py")
    spec = importlib.util.spec_from_file_location("streamlit_app", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # __name__ != "__main__" -> main() does not run
    return mod


def test_gui_smoke():
    try:
        import streamlit  # noqa: F401
    except Exception:
        print("SKIP: streamlit not installed (core install) -- GUI smoke skipped")
        return

    app = _load_app_module()
    base = os.path.join(ROOT, "examples")
    npz = os.path.join(base, "xxz_Sx_integrable.npz")

    # End-to-end on the bundled npz example.
    res = app.analyze(npz)
    assert res["report"].regime is not None
    assert isinstance(res["summary"], str) and "regime" in res["summary"]
    assert isinstance(res["dashboard"], Figure)
    assert isinstance(res["stateflow"]["figure"], Figure)
    assert res["info"]["dimensionality"] == 1
    assert res["stateflow"]["n_episodes"] >= 0
    ref = res["report"]

    # The GUI's loader path must agree with the npz path for .mat and .csv,
    # proving the uploader -> load_any -> classify pipeline is consistent.
    from scipy.io import savemat
    kn = res["kernel"]
    tmp = tempfile.mkdtemp()

    matp = os.path.join(tmp, "k.mat")
    savemat(matp, {"K": kn.K.T, "t": kn.t, "r": kn.r})  # store (R, T) on purpose
    rmat = app.analyze(matp)
    assert rmat["report"].regime == ref.regime
    assert rmat["report"].coherence == ref.coherence

    csvp = os.path.join(tmp, "k_long.csv")
    rows = [[kn.t[i], kn.r[j], kn.K[i, j]]
            for i in range(kn.K.shape[0]) for j in range(kn.K.shape[1])]
    np.savetxt(csvp, np.array(rows), delimiter=",")
    rcsv = app.analyze(csvp)
    assert rcsv["report"].regime == ref.regime
    assert rcsv["report"].coherence == ref.coherence

    _render_check()
    print("ALL GUI TESTS PASS")


def _render_check():
    """Genuine headless render check: run the Streamlit script via the official
    AppTest harness, exercising main() and every st.* call without a browser."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception:
        print("SKIP: streamlit.testing.AppTest unavailable -- render check skipped")
        return
    app_path = os.path.join(ROOT, "app", "streamlit_app.py")
    at = AppTest.from_file(app_path, default_timeout=60).run()
    assert not at.exception, at.exception
    assert len(at.info) >= 1, "initial page should prompt for input"

    # Click the demo button -> full pipeline renders with no exception/error.
    assert len(at.button) >= 1, "demo button missing"
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert len(at.error) == 0, f"unexpected st.error: {[e.value for e in at.error]}"
    assert any("regime" in c.value for c in at.code), "regime summary missing"
    assert len(at.metric) >= 4, "kernel-info + stateflow metrics expected"

    # Changing a sidebar knob must re-run cleanly.
    at.number_input[0].set_value(0.10).run()
    assert not at.exception and len(at.error) == 0
    print("render check OK (AppTest: demo pipeline + param change, no exceptions)")


if __name__ == "__main__":
    test_gui_smoke()
