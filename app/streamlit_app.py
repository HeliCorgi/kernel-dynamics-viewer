"""Streamlit GUI for kernel-dynamics-viewer (v0.9.2 polish over the v0.8 app).

Drag in a kernel file (npz/npy/mat/csv) -> automatic full analysis -> a clean
dashboard showing (i) the kernel/regime summary, (ii) the stateflow axis
timelines (continuous transport slope and coherence E(t)), and (iii) the
transitions.

This module is a *thin shell* over existing library functions, exactly like the
CLI. It contains NO new analysis math: it wires together
``kernel_viewer.io.loader.load_any`` -> ``kernel_viewer.regimes.classify_any`` ->
the existing matplotlib dashboards and the ``stateflow`` axes. All the real work
lives in the library.

The analysis wiring is exposed as plain importable functions (``load_capturing``,
``kernel_info``, ``compute_regime``, ``make_dashboard``, ``compute_stateflow``,
``analyze``, plus the ``report_json_bytes`` / ``dashboard_pdf_bytes`` export
helpers) so it can be smoke-tested headlessly without launching a browser.
Every Streamlit call lives inside ``main()``; importing this module runs no UI.

v0.9.2 polish over the v0.8 app: a sidebar "Load Example Dataset" dropdown
auto-listed from examples/*.npz, ``st.spinner`` progress on the slow steps, and a
JSON (reusing ``save_report``) + single-PDF (the shown dashboard) download area.

Run it with::

    pip install -e ".[gui]"
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import warnings

import matplotlib
matplotlib.use("Agg")  # headless-safe; Streamlit renders the Figure object
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Make the package importable when run via `streamlit run app/streamlit_app.py`
# (the app/ dir is not part of the installed package).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from kernel_viewer.io.loader import load_any
from kernel_viewer.io.saver import save_report
from kernel_viewer.regimes.classifier import classify_any
from kernel_viewer.core.kernel2d import Kernel2D
from kernel_viewer.core.kernel3d import Kernel3D
from kernel_viewer.core.kernel4d import Kernel4D
from kernel_viewer.visualization.dashboard import plot_dashboard
from kernel_viewer.visualization.dashboard2d import plot_dashboard2d
from kernel_viewer.visualization.dashboard3d import plot_dashboard3d
from kernel_viewer.visualization.dashboard4d import plot_dashboard4d

from stateflow import KernelExtractor, TransportAxis, CoherenceAxis
from stateflow import viz as sviz

EXAMPLES_DIR = os.path.join(_REPO_ROOT, "examples")
UPLOAD_TYPES = ["npz", "npy", "mat", "csv", "txt"]

# Library defaults, surfaced read-only in the sidebar (no auto-tuning).
DEFAULT_RING_THRESHOLD = 0.05
DEFAULT_EDGE_LIMIT = 0.3


# ---------------------------------------------------------------------------
# Importable analysis wiring (Streamlit-free; safe to call from tests).
# ---------------------------------------------------------------------------

def list_examples():
    """Bundled example .npz files, sorted by name."""
    if not os.path.isdir(EXAMPLES_DIR):
        return []
    return sorted(f for f in os.listdir(EXAMPLES_DIR) if f.endswith(".npz"))


def load_capturing(path, time_axis="auto", layout="auto"):
    """Call load_any while capturing any loader warnings (e.g. the .mat
    transpose note) so the GUI can surface them. Returns (kernel, warnings)."""
    ta = None if time_axis in (None, "auto") else time_axis
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kernel = load_any(path, time_axis=ta, layout=layout)
    return kernel, [str(w.message) for w in caught]


def kernel_dim(kernel):
    return {"Kernel": 1, "Kernel2D": 2, "Kernel3D": 3, "Kernel4D": 4}.get(
        type(kernel).__name__, 1
    )


def kernel_info(kernel):
    """Human-facing summary of the loaded kernel's shape and time span."""
    t = kernel.t
    return {
        "type": type(kernel).__name__,
        "dimensionality": kernel_dim(kernel),
        "K_shape": tuple(int(s) for s in kernel.K.shape),
        "n_times": int(len(t)),
        "t_min": float(t[0]),
        "t_max": float(t[-1]),
    }


def compute_regime(kernel, ring_threshold=DEFAULT_RING_THRESHOLD,
                   edge_limit=DEFAULT_EDGE_LIMIT):
    """The single source of truth for the regime: kernel_viewer.classify_any."""
    return classify_any(kernel, ring_threshold=ring_threshold,
                        edge_limit=edge_limit)


def make_dashboard(kernel, report):
    """Dimension-appropriate matplotlib dashboard Figure (path=None)."""
    if isinstance(kernel, Kernel4D):
        return plot_dashboard4d(kernel, report)
    if isinstance(kernel, Kernel3D):
        return plot_dashboard3d(kernel, report)
    if isinstance(kernel, Kernel2D):
        return plot_dashboard2d(kernel, report)
    return plot_dashboard(kernel, report)


def compute_stateflow(kernel, ring_threshold=DEFAULT_RING_THRESHOLD):
    """Build the stateflow panel: a transport axis (continuous log-log slope,
    scalar exponent p) and a coherence axis (continuous E(t), ring episodes),
    plus the four-panel stateflow dashboard Figure and the transition list.

    This calls the unmodified stateflow API; it computes no new math."""
    fs = KernelExtractor().from_kernel(kernel)
    transport = TransportAxis().compute(fs)
    coherence = CoherenceAxis(ring_threshold=ring_threshold).compute(fs)
    figure = sviz.dashboard(transport, coherence)
    transitions = [
        {
            "time": float(getattr(tr, "time", float("nan"))),
            "kind": getattr(tr, "kind", ""),
            "E_max": float(getattr(tr, "detail", {}).get("E_max", float("nan"))),
        }
        for tr in getattr(coherence, "transitions", [])
    ]
    return {
        "figure": figure,
        "transport_p": float(getattr(transport, "scalar", float("nan"))),
        "t_clean": float(getattr(transport, "t_clean", float("nan"))),
        "n_episodes": int(len(getattr(coherence, "episodes", []))),
        "transitions": transitions,
    }


def analyze(path, ring_threshold=DEFAULT_RING_THRESHOLD,
            edge_limit=DEFAULT_EDGE_LIMIT, time_axis="auto", layout="auto"):
    """End-to-end pipeline used by the headless GUI smoke test: load -> classify
    -> dashboard -> stateflow. Raises on any failure (no silent swallowing)."""
    kernel, warns = load_capturing(path, time_axis=time_axis, layout=layout)
    report = compute_regime(kernel, ring_threshold, edge_limit)
    return {
        "kernel": kernel,
        "warnings": warns,
        "info": kernel_info(kernel),
        "report": report,
        "summary": report.summary(),
        "dashboard": make_dashboard(kernel, report),
        "stateflow": compute_stateflow(kernel, ring_threshold),
    }


def _write_upload(uploaded):
    """Persist a Streamlit UploadedFile to a temp path (load_any needs a path)."""
    suffix = os.path.splitext(uploaded.name)[1] or ".dat"
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded.getbuffer())
    return tmp


def report_json_bytes(report):
    """JSON bytes for a RegimeReport, produced by reusing the library's
    ``kernel_viewer.io.saver.save_report`` -- the single source of truth for the
    report dict (regime / coherence / exponent / features / notes). Not
    hand-serialized."""
    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        save_report(report, tmp)
        with open(tmp, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def dashboard_pdf_bytes(fig):
    """Render the already-built dashboard ``Figure`` to a one-page PDF (bytes)
    via matplotlib's own pdf backend -- no new dependency, no multi-page
    assembly, just the single figure the app is already showing."""
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Streamlit UI (only runs under `streamlit run`, never on import).
# ---------------------------------------------------------------------------

def main():
    import streamlit as st

    st.set_page_config(page_title="kernel-dynamics-viewer", layout="wide")
    st.title("kernel-dynamics-viewer")
    st.caption(
        "Drag in a spatiotemporal kernel K(r, t) and get its transport/coherence "
        "regime, the matplotlib dashboard, and the stateflow axis timelines. "
        "A thin shell over the library — no analysis math lives here."
    )

    # --- Sidebar: the real, existing knobs (library defaults, user override). ---
    st.sidebar.header("Parameters")
    ring_threshold = st.sidebar.number_input(
        "ring_threshold (coherence)", min_value=0.0, max_value=1.0,
        value=DEFAULT_RING_THRESHOLD, step=0.01, format="%.3f",
        help="Ring-strength threshold E* separating coherent episodes. "
             "Passed to classify_any and the stateflow CoherenceAxis.",
    )
    edge_limit = st.sidebar.number_input(
        "edge_limit (clean window)", min_value=0.0, max_value=1.0,
        value=DEFAULT_EDGE_LIMIT, step=0.05, format="%.2f",
        help="Edge-fraction cutoff defining the boundary-clean window for the "
             "transport-exponent fit. Passed to classify_any.",
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("Loader hints (.mat / .csv)")
    time_axis = st.sidebar.selectbox(
        "mat time_axis", ["auto", "first", "last"], index=0,
        help="Override MATLAB time-axis detection ((R, T) vs (T, R)).",
    )
    layout = st.sidebar.selectbox(
        "csv layout", ["auto", "matrix", "long"], index=0,
        help="matrix: (T, R) grid (NaN corner = labelled). long: columns t, r, K.",
    )

    # --- Sidebar: load a bundled example (auto-listed from examples/*.npz). ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Load Example Dataset")
    examples = list_examples()
    chosen = st.sidebar.selectbox(
        "examples/*.npz", examples,
        index=examples.index("xxz_Sx_integrable.npz")
        if "xxz_Sx_integrable.npz" in examples else 0,
        help="Auto-listed from the examples/ folder.",
    ) if examples else None
    demo_clicked = st.sidebar.button(
        "Load Example", disabled=not examples, use_container_width=True)

    # --- Main: file uploader. ---
    st.subheader("Input")
    up = st.file_uploader(
        "Upload a kernel file", type=UPLOAD_TYPES,
        help="npz/npy (K[, r, t]), mat (K + coords), or 1D csv/txt. "
             "Or pick a bundled example in the sidebar.",
    )

    # Resolve the active path. Upload wins; the example selection persists across
    # reruns (e.g. when a slider changes) via session_state.
    path, source = None, None
    if up is not None:
        path = _write_upload(up)
        source = up.name
    elif demo_clicked and chosen:
        st.session_state["demo_path"] = os.path.join(EXAMPLES_DIR, chosen)
        st.session_state["demo_name"] = chosen
    if path is None and "demo_path" in st.session_state:
        path = st.session_state["demo_path"]
        source = st.session_state.get("demo_name", os.path.basename(path))

    if path is None:
        st.info("Upload a kernel file or load a bundled example to begin.")
        return

    st.markdown(f"**Source:** `{source}`")

    # --- Load (spinner + graceful errors). ---
    try:
        with st.spinner("Loading kernel..."):
            kernel, warns = load_capturing(path, time_axis=time_axis, layout=layout)
    except NotImplementedError as e:
        st.error(f"Unsupported input: {e}")
        return
    except Exception as e:  # noqa: BLE001 - surface everything, never blank-page
        st.error(f"Could not load the file: {e}")
        return
    for w in warns:
        st.warning(w)

    info = kernel_info(kernel)
    m = st.columns(4)
    m[0].metric("dimensionality", f"{info['dimensionality']}D")
    m[1].metric("K shape", "x".join(str(s) for s in info["K_shape"]))
    m[2].metric("times", info["n_times"])
    m[3].metric("t range", f"[{info['t_min']:g}, {info['t_max']:g}]")

    # --- Regime panel: exact CLI summary text + matplotlib dashboard. ---
    st.subheader("Regime")
    report, pdf_bytes = None, None
    try:
        with st.spinner("Classifying transport x coherence..."):
            report = compute_regime(kernel, ring_threshold, edge_limit)
        st.code(report.summary(), language="text")
        with st.spinner("Rendering dashboard..."):
            fig = make_dashboard(kernel, report)
        st.pyplot(fig)
        pdf_bytes = dashboard_pdf_bytes(fig)   # capture the SHOWN figure for PDF
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        st.error(f"Classification / dashboard failed: {e}")
        return

    # --- Downloads: JSON via save_report (reused), PDF of the shown dashboard. ---
    st.subheader("Download results")
    d1, d2 = st.columns(2)
    try:
        d1.download_button(
            "Download report (JSON)", report_json_bytes(report),
            file_name="kernel_report.json", mime="application/json")
    except Exception as e:  # noqa: BLE001
        d1.error(f"JSON export failed: {e}")
    if pdf_bytes is not None:
        d2.download_button(
            "Download dashboard (PDF)", pdf_bytes,
            file_name="kernel_dashboard.pdf", mime="application/pdf")

    # --- stateflow panel: continuous axes + episodes + transitions. ---
    st.subheader("stateflow axes")
    try:
        with st.spinner("Computing axis trajectories..."):
            sf = compute_stateflow(kernel, ring_threshold)
        s1, s2 = st.columns(2)
        p = sf["transport_p"]
        s1.metric("transport exponent p", "n/a" if p != p else f"{p:.3f}")
        s2.metric("E(t) ring episodes", sf["n_episodes"])
        st.pyplot(sf["figure"])
        plt.close(sf["figure"])
        if sf["transitions"]:
            st.markdown("**Transitions**")
            st.table({
                "time": [t["time"] for t in sf["transitions"]],
                "kind": [t["kind"] for t in sf["transitions"]],
                "E_max": [t["E_max"] for t in sf["transitions"]],
            })
        else:
            st.write("No coherence transitions detected.")
    except Exception as e:  # noqa: BLE001
        st.error(f"stateflow panel failed: {e}")

    st.markdown("---")
    st.caption(
        "Planned: Plotly interactivity, K(r,t) time-slider animation, "
        "multi-file comparison."
    )


if __name__ == "__main__":
    main()
