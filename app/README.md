# kernel-dynamics-viewer GUI (Streamlit)

A thin drag-and-drop front-end over the existing `kernel_viewer` and `stateflow`
libraries. It contains **no analysis math** — it only wires together
`load_any` → `classify_any` → the matplotlib dashboards and the stateflow axes,
exactly like the CLI.

```bash
pip install -e ".[gui]"             # Streamlit is an optional extra
streamlit run app/streamlit_app.py
```

## Scope (v0.8 — "medium")

Upload a file (or load a bundled example) → automatic full analysis → one clean
page with three parts:

1. **Loaded kernel** — detected dimensionality, `K` shape, time range, and any
   loader warning (e.g. the `.mat` time-axis transpose note).
2. **Regime** — the exact `classify_any(...).summary()` text the CLI prints,
   plus the dimension-appropriate matplotlib dashboard (`plot_dashboard*`).
3. **stateflow axes** — the continuous transport-slope and coherence `E(t)`
   timelines (`stateflow.viz.dashboard`), the transport exponent `p` and the
   number of `E(t)` episodes as metrics, and the transitions in a table.

The sidebar exposes only the two **real, existing** knobs — `ring_threshold`
(0.05) and `edge_limit` (0.3) — at their library defaults, passed through to
`classify_any` and the stateflow `CoherenceAxis`. Nothing is auto-tuned and no
new parameters are invented. Load/analyze errors are surfaced with `st.error`
rather than crashing to a blank page.

## Architecture

All analysis wiring lives in plain, importable functions (`load_capturing`,
`compute_regime`, `make_dashboard`, `compute_stateflow`, `analyze`). Every
Streamlit call lives inside `main()`, which only runs under `streamlit run` —
importing the module runs no UI, so the pipeline can be smoke-tested headlessly
(`tests/test_gui_smoke.py`) without launching a browser. The module is never
imported by the core package or the core test suite, so the core install stays
`numpy + scipy + matplotlib`.

## Explicitly deferred (not in v0.8)

These are the **next** milestone, deliberately left out so v0.8 ships:

- **Plotly interactivity** (zoom/hover on the dashboards).
- **`K(r, t)` time-slider animation** of the kernel profile.
- **Multi-file comparison** (load several kernels side by side).

Further out, and out of scope here: HDF5 input, QuTiP/ITensor/TEBD exporters,
"pull data from a link" acquisition workflows, and uncertainty bootstrapping.
