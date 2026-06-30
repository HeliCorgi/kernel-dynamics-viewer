import numpy as np
import matplotlib.pyplot as plt


def plot_heatmap(kernel, ax=None, log=False):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    A = np.abs(kernel.K) if log else kernel.K
    if log:
        A = np.log10(np.clip(A, kernel.noise_floor() * 0.1, None))
    im = ax.pcolormesh(kernel.r, kernel.t, A, shading="auto", cmap="magma")
    ax.set_xlabel("r"); ax.set_ylabel("t")
    ax.set_title(("log10 " if log else "") + "K(r, t)")
    plt.colorbar(im, ax=ax)
    return ax
