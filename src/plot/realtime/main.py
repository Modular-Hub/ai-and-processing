import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def generate_data():
    x = 0
    while True:
        yield x, np.sin(x)
        x += 0.1

fig, ax = plt.subplots()
xdata, ydata = [], []
line, = ax.plot([], [], 'b-')

def init():
    ax.set_ylim(-1, 1)
    return line,

def animate(data):
    x, y = data
    xdata.append(x)
    ydata.append(y)
    line.set_data(xdata, ydata)

    ax.set_xlim(x - 5, x + 5)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: int(val)))
    ax.figure.canvas.flush_events()

    return line,

ani = FuncAnimation(fig, animate, generate_data, init_func=init, interval=50, cache_frame_data=False)
plt.show()
