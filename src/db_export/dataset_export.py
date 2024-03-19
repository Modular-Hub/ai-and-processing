from os import path
import json


DATASET_DATE = "2024-02-17"
DATASET_ELECTRODES = "t6-c4" # t5-c3, t6-c4
DATASET_CATEGORY = "SI"
DATASET_BOARD_VERSION = ["v1", "v2", "v3"]
DATASET_PATH = f"{path.abspath('.')}/../data/{DATASET_DATE}/{DATASET_ELECTRODES}"


# Read file
with open(f"{DATASET_PATH}/{DATASET_CATEGORY}.csv", 'r') as file:
    lines = file.readlines()


# Database object
obj = {
    "date":         DATASET_DATE,
    "electrodes":   DATASET_ELECTRODES,
    "category":     DATASET_CATEGORY,
    "board_version":DATASET_BOARD_VERSION[1],
    "number_of_proofs": 0,
    "proofs": []
}

# Split data and fix format
data = []
for line in lines:
    values = [int(val.strip()) for val in line.split(',') if val.strip()]    
    if len(values) > 0:
        data.append(values)

obj["number_of_proofs"] = len(data)
obj["proofs"] = data

with open(f"{DATASET_PATH}/{DATASET_CATEGORY}_export.json", "w") as file:
    json.dump(obj, file, indent=4)