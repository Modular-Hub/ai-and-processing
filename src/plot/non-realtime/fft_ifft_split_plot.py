from scipy.fft import fft, fftfreq, ifft
import matplotlib.pyplot as plt
import numpy as np
from os import mkdir, path

SAVE_FIGS_FILES = 0b11
NPROOFS = 20
FOLDER = f"{path.abspath(".")}/../serial comunication/2024-01-13/t3-c5/SI/"

# File handler
def writeInFile(path, data):
    with open(path, "+a") as file:
        file.write(f"{data}\n")

# FFT Results subfolder
fft_plot_path = f"{FOLDER}fft_plot_results/"
try:
    mkdir(fft_plot_path)
except OSError as error:
    print(error)

# IFFT Results subfolder
ifft_plot_path = f"{FOLDER}ifft_plot_results/"
try:
    mkdir(ifft_plot_path)
except OSError as error:
    print(error)

# FFT data subfolder
fft_path = f"{FOLDER}fft_data/"
try:
    mkdir(fft_path)
except OSError as error:
    print(error)

# IFFT data subfolder
ifft_path = f"{FOLDER}ifft_data/"
try:
    mkdir(ifft_path)
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
N = data.shape[1]
T = 1 / N
X_FFT_FREQ = fftfreq(N, T)[:N//2]
FREQ_RANGES = [(0, 2), (3, 5), (6, 12), (13, 33), (24, 90), (91, 245), (245, N-1)]
LEN_FREQ = 7

# FFT & IFFT
fftData = np.zeros((NPROOFS, N))
ifftData = np.zeros((NPROOFS, N))
fftDataFreqSplit = np.zeros((NPROOFS, LEN_FREQ, N))
ifftDataFreqSplit = np.zeros((NPROOFS, LEN_FREQ, N))
for i in range(NPROOFS):
    fftData[i][1:480] = fft(data[i])[1:480]
    ifftData[i] = ifft(fftData[i])

    # FFT split & IFFT calculation
    for j, freqRange in enumerate(FREQ_RANGES):
        left, right = freqRange
        fftDataFreqSplit[i][j][left:right] = fftData[i][left:right]
        ifftDataFreqSplit[i][j] = ifft(fftDataFreqSplit[i][j])


# Plots & Files
for idx_proof in range(NPROOFS):
    figFFT, axsFFT = plt.subplots(LEN_FREQ)
    figIFFT, axsIFFT = plt.subplots(LEN_FREQ)

    # Plot by Freq
    for idx_freq in range(LEN_FREQ):
        # FFT
        axsFFT[idx_freq].plot(X_FFT_FREQ, 2.0/N*np.abs(fftDataFreqSplit[idx_proof][idx_freq][0:N//2]))
    
        # IFFT
        axsIFFT[idx_freq].plot(ifftDataFreqSplit[idx_proof][idx_freq])

        # Files save
        if SAVE_FIGS_FILES & 0b01:
            writeInFile(f"{fft_path}proof-{idx_proof + 1}.csv", list(fftDataFreqSplit[idx_proof][idx_freq]))
            writeInFile(f"{ifft_path}proof-{idx_proof + 1}.csv", list(ifftDataFreqSplit[idx_proof][idx_freq]))

    # Figs save
    if SAVE_FIGS_FILES & 0b10:
        figFFT.savefig(f"{fft_plot_path}proof-{idx_proof + 1}.jpg", dpi=400)
        figIFFT.savefig(f"{ifft_plot_path}proof-{idx_proof + 1}.jpg", dpi=400)
    
    plt.close(figFFT)
    plt.close(figIFFT)