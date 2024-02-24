import matplotlib.pyplot as plt
import numpy as np
import pywt

MAX_RANGE = 256
data = np.fromfile("./EEG_test_data.csv", dtype=int, sep=",")
coef, freq = pywt.cwt(data, np.arange(1, MAX_RANGE), "gaus1")

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
plt.savefig("test.jpg", bbox_inches="tight", dpi=500)
