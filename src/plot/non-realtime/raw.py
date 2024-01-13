import matplotlib.pyplot as plt
import numpy as np

ydata = []
for i in range(1, 6):
    ydata.append(np.genfromtxt(f"{i}.data", dtype=int, delimiter=","))


for i in range(0, 5):
    print(len(ydata[i]), end="\t")
    data_max = max(ydata[i])
    print(data_max, end="\t")
    print(len([i for i in ydata[i] if i==data_max]))

