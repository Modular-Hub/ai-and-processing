import matplotlib.pyplot as plt
import numpy as np
from pywt import cwt


def spectrogram_save(data, n_proof, path):
    max_range = 256
    data[0:10] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    coef, _ = cwt(data, np.arange(1, max_range), "morl")

    # Generate frame
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="both", which="both", length=0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Plot
    ax.imshow(
        coef,
        extent=[-1, 1, 1, max_range],
        interpolation="bilinear",
        cmap="binary", #ESCALA DE GRISES
        aspect="auto",
        vmax=abs(coef).max(),
        vmin=-abs(coef).max(),
    )

    # Save fig
    plt.savefig(f"{path}/{n_proof}.jpg", bbox_inches="tight", dpi=500)
    plt.close(fig)
