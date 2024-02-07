import numpy as np
import random
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
import csv
import json

# Use StratifiedShuffleSplit to split the data while maintaining the class distribution
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) # random_state=42 for reproducibility

labels = []
# Open the SA_data.csv file (SST text)
with open('ZAB_SA_data.csv', 'r') as infile:
    reader = csv.reader(infile)
    header_row = next(reader) # Skip the header row in the reader
    for i, row in enumerate(reader):
        label = row[2]
        labels.append(label)
print(f'labels: {labels}')
print(f'len(labels): {len(labels)}')


# Generate stratified indices for train and test sets for balanced class distribution
train_indices = []
test_indices = []
for train_index, test_index in sss.split(range(len(labels)), labels):
    train_indices.extend(train_index)
    test_indices.extend(test_index)

indices = {
    "train_indices": train_indices,
    "test_indices": test_indices
}

print(f'inidices: {indices}')

with open('kfold_splitting_indices.pkl', 'wb') as file:
    pickle.dump(indices, file)

