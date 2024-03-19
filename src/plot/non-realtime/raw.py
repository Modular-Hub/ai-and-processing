import matplotlib.pyplot as plt
import numpy as np
from os import path


NPROOFS = 2
# FOLDER = f"{path.abspath(".")}/../serial comunication/2024-01-13/t6-c4/NO/"
FOLDER = f"./test.csv"


# Read test file
with open(f'test.csv', 'r') as file:
    lines = file.readlines()

# Split data and fix format
data = []
for line in lines:
    values = [int(val.strip()) for val in line.split(',') if val.strip()]    
    if len(values) > 0:
        data.append(np.array(values))

data = np.vstack(data)
N = data.shape[1]

## 4000
x1, y1 = [0, 2000], [4000, 4000]

plt.plot(x1, y1, marker = 'o')

for idx, d in enumerate(data):
    print(max(d))
    plt.plot(d, label=f"line{idx+1}")

plt.legend()
plt.show()