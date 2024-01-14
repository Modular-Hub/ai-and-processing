from scipy.fft import fft, fftfreq, ifft
import matplotlib.pyplot as plt
import numpy as np
from os import mkdir

NPROOFS = 20
FOLDER = "/Users/felipejim/Desktop/Modular/ia-and-processing/src/plot/serial comunication/2024-01-13/t3-c5/SI/"

# Results subfolder
plot_path = f"{FOLDER}plot_results/"
try:
    mkdir(plot_path)
except OSError as error:
    print(error)

# Read test file
with open(f'{FOLDER}test.csv', 'r') as file:
    lines = file.readlines()

# Split data and fix format
data = []
for line in lines:
    values = [int(val.strip()) for val in line.split(',') if val.strip()]    
    if len(values) > 0:
        data.append(np.array(values))

data = np.vstack(data)

# FFT
N = data.shape[1]
T = 1 / N
fftData = []
xfftFreq = fftfreq(N, T)[:N//2]
for i in range(NPROOFS):
    fftResult = fft(data[i])
    fftData.append(np.zeros(len(fftResult)))
    fftData[i][1:480] = fftResult[1:480]

# Plots
FREQ_RANGES = [(0, 2), (3, 5), (6, 12), (13, 33), (24, 90), (91, 245), (245, 1)]
LEN_FREQ = 7
YLEN = len(fftData[0])
FREQ_RANGES[-1] = (245, YLEN - 1)
for idx_proof in range(NPROOFS):
    # Split FFT by frequencies & plot it
    fig, axs = plt.subplots(LEN_FREQ)
    yFreqSeparated = np.zeros((LEN_FREQ, YLEN))
    for idx in range(LEN_FREQ):
        left, right = FREQ_RANGES[idx]
        yFreqSeparated[idx][left:right] = fftData[idx_proof][left:right]
        axs[idx].plot(xfftFreq, 2.0/N*np.abs(yFreqSeparated[idx][0:N//2]))

    fig.savefig(f"{plot_path}proof-{idx_proof + 1}.jpg")
    

# Plot Inverse FFT
# fig2, axs2 = plt.subplots(1)
# left = 1000
# invfft = ifft(yfdata[0])
# axs2.plot(inv fft)

# plt.show()
