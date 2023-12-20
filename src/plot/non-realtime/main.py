from scipy.fft import fft, fftfreq, ifft
import matplotlib.pyplot as plt
import numpy as np

ydata = []
for i in range(1, 6):
    ydata.append(np.genfromtxt(f"{i}.data", dtype=int, delimiter=","))

# Buffer allocation fix
ydata[0] = ydata[0][253:]

# FFT
N = 25001
T = 1 / N
yfdata = []
xf = []
for i in range(0, 5):
    yfdata.append(fft(ydata[i]))
    xf.append(fftfreq(N, T)[:N//2])

    xf[i] = xf[i][2000:8000]
    yfdata[i] = yfdata[i][2000:8000]



# Plot FFT direct
fig, axs = plt.subplots(5)
for i in range(0, 5):
    # print(min(ydata[i]), max(ydata[i]), len(ydata[i]))
    # axs[i].plot(fft(ydata[i]), fftfreq())
    axs[i].plot(xf[i], 2.0/N*np.abs(yfdata[i][0:N//2]))


# Plot Inverse FFT
fig2, axs2 = plt.subplots(6)
left = 1000
for i in range(0, 6):
    invfft = ifft(yfdata[0][left*i:(left + (1000 * i) - 1)])
    axs2[i].plot(invfft)


plt.show()
