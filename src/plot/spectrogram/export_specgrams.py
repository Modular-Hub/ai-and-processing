from os import path, mkdir
import numpy as np
import specgram

ELEC_COMB = ["t5-c3", "t6-c4", "f7-t3"]
TEST_TYPE = "5"
DATA_PATH = f"{path.abspath('.')}/../../data/2024-03-23/{ELEC_COMB[2]}"
FILE_PATH = f"{DATA_PATH}/{TEST_TYPE}.csv"

# Images sub folder
IMGS_PATH = f"{DATA_PATH}/{TEST_TYPE}_imgs"
try:
    mkdir(IMGS_PATH)
except OSError as error:
    print(error)

# Read test file
with open(FILE_PATH, "r") as file:
    lines = file.readlines()

# Split data and fix format
data = []
for line in lines:
    values = [int(val.strip()) for val in line.split(",") if val.strip()]
    if len(values) > 0:
        data.append(np.array(values))

data = np.vstack(data)

for n, proof in enumerate(data):
    specgram.spectrogram_save(proof, (n + 1 + 0), IMGS_PATH)
