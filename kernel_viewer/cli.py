import argparse
import sys
import numpy as np
from .io.loader import load_any
from .io.saver import save_report
from .regimes.classifier import classify_any
from .regimes import models, models2d, models3d, models4d
from .core.kernel2d import Kernel2D
from .core.kernel3d import Kernel3D
from .core.kernel4d import Kernel4D
from .core.spectral import compute_spectrum_radial
from .visualization.dashboard import plot_dashboard
from .visualization.dashboard2d import plot_dashboard2d
from .visualization.dashboard3d import plot_dashboard3d
from .visualization.dashboard4d import plot_dashboard4d
from .visualization.ring_plot import plot_ring
from .visualization.spectrum import plot_spectrum
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(argv=None):
    p = argparse.ArgumentParser(prog="kernel-viewer",
                                description="Regime diagnostics for spatiotemporal kernels K(r,t)")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("analyze", "classify", "plot-ring", "spectrum"):
        sp = sub.add_parser(name)
        sp.add_argument("data", help=".npz/.npy (K[, r, t]), .mat (K + coords), or 1D .csv/.txt")
        sp.add_argument("-o", "--out", default=None)
    sp = sub.add_parser("track", help="sliding-window regime tracking (transition detection)")
    sp.add_argument("data")
    sp.add_argument("--window", type=int, required=True, help="frames per window")
    sp.add_argument("--stride", type=int, default=1)
    sub.add_parser("demo", help="generate synthetic 1D kernels and analyze them")
    sub.add_parser("demo2d", help="generate synthetic/simulated 2D kernels and analyze them")
    sub.add_parser("demo3d", help="generate synthetic/simulated 3D kernels and analyze them")
    sub.add_parser("demo4d", help="generate synthetic/simulated 4D kernels and analyze them")
    args = p.parse_args(argv)

    if args.cmd == "demo":
        for name in ("diffusive", "ballistic", "levy", "recurrent"):
            kn = getattr(models, name)()
            rep = classify_any(kn)
            np.savez(f"demo_{name}.npz", r=kn.r, t=kn.t, K=kn.K)
            plot_dashboard(kn, rep, path=f"demo_{name}.png")
            print(f"== {name} ==\n{rep.summary()}\n-> demo_{name}.png\n")
        return 0
    if args.cmd == "demo4d":
        for name in ("diffusive4d", "ballistic4d", "levy4d", "cubic4d", "randomwalk4d"):
            kn = getattr(models4d, name)()
            rep = classify_any(kn)
            plot_dashboard4d(kn, rep, path=f"demo_{name}.png")
            print(f"== {name} ==\n{rep.summary()}\n-> demo_{name}.png\n")
        return 0
    if args.cmd == "demo3d":
        for name in ("diffusive3d", "ballistic3d", "levy3d", "cubic3d", "randomwalk3d"):
            kn = getattr(models3d, name)()
            rep = classify_any(kn)
            plot_dashboard3d(kn, rep, path=f"demo_{name}.png")
            print(f"== {name} ==\n{rep.summary()}\n-> demo_{name}.png\n")
        return 0
    if args.cmd == "demo2d":
        for name in ("diffusive2d", "ballistic2d", "levy2d", "anisotropic2d", "randomwalk2d", "wave2d"):
            kn = getattr(models2d, name)()
            rep = classify_any(kn)
            np.savez(f"demo_{name}.npz", x=kn.x, y=kn.y, t=kn.t, K=kn.K.astype(np.float32))
            plot_dashboard2d(kn, rep, path=f"demo_{name}.png")
            print(f"== {name} ==\n{rep.summary()}\n-> demo_{name}.png\n")
        return 0

    kn = load_any(args.data)
    if args.cmd == "track":
        from .metrics.tracking import track_regimes
        for r in track_regimes(kn, window=args.window, stride=args.stride):
            line = (f"t=[{r['t_start']:g},{r['t_end']:g}]  {r['regime']:<14} "
                    f"{r['coherence']:<16}")
            print(line + ("  <-- TRANSITION" if r["transition"] else ""))
        return 0
    rep = classify_any(kn)
    if args.cmd == "classify":
        print(rep.summary())
        if args.out:
            save_report(rep, args.out)
            print(f"report -> {args.out}")
    elif args.cmd == "analyze":
        out = args.out or "dashboard.png"
        if isinstance(kn, Kernel4D):
            plot_dashboard4d(kn, rep, path=out)
        elif isinstance(kn, Kernel3D):
            plot_dashboard3d(kn, rep, path=out)
        elif isinstance(kn, Kernel2D):
            plot_dashboard2d(kn, rep, path=out)
        else:
            plot_dashboard(kn, rep, path=out)
        print(rep.summary())
        print(f"dashboard -> {out}")
    elif args.cmd in ("plot-ring", "spectrum") and isinstance(kn, (Kernel2D, Kernel3D, Kernel4D)):
        rk = kn.radial()
        if args.cmd == "spectrum":
            dim = rk.geometry_dim
            iso = rep.features.get("isotropy_score", None)
            sp = compute_spectrum_radial(rk, dim=dim)
            from .visualization.spectrum import plot_spectrum
            plot_spectrum(rk, spectrum=sp)
            out = args.out or "spectrum.png"
            plt.gca().set_title(f"|S(k, $\\omega$)| via {'Hankel J0' if dim == 2 else 'spherical Bessel j0'} "
                                f"(assumes isotropy; isotropy_score={iso})", fontsize=9)
            plt.savefig(out, dpi=130, bbox_inches="tight")
            print(f"spectrum ({dim}D radial, isotropy assumed) -> {out}")
            if iso is not None and iso < 0.9:
                print(f"warning: isotropy_score={iso} < 0.9 -- the radial S(k,omega) mixes directions")
        else:
            out = args.out or "ring.png"
            plot_ring(rk)
            plt.savefig(out, dpi=130, bbox_inches="tight")
            print(f"ring plot (radial profile) -> {out}")
    elif args.cmd == "plot-ring":
        out = args.out or "ring.png"
        plot_ring(kn)
        plt.savefig(out, dpi=130, bbox_inches="tight")
        print(f"ring plot -> {out}")
    elif args.cmd == "spectrum":
        out = args.out or "spectrum.png"
        plot_spectrum(kn)
        plt.savefig(out, dpi=130, bbox_inches="tight")
        print(f"spectrum -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
