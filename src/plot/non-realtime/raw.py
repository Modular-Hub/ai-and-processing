import matplotlib.pyplot as plt
import numpy as np
from os import path

SAVE_FIGS_FILES = 0b11
NPROOFS = 20
FOLDER = f"{path.abspath(".")}/../serial comunication/2024-01-13/t6-c4/NO/"


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

plt.plot(data[5])
plt.show()