import matplotlib.pyplot as plt
import numpy as np
import pywt


def specgram_save(data, n_proof, path):
    MAX_RANGE = 256
    coef, _ = pywt.cwt(data, np.arange(1, MAX_RANGE), "gaus1")

    ## Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="both", which="both", length=0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    ax.imshow(
        coef,
        extent=[-1, 1, 1, MAX_RANGE],
        interpolation="bilinear",
        cmap="plasma",
        aspect="auto",
        vmax=abs(coef).max(),
        vmin=-abs(coef).max(),
    )
    plt.savefig(f"{path}/{n_proof}.jpg", bbox_inches="tight", dpi=500)
    plt.close(fig)
