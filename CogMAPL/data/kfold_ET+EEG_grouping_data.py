import torch
import numpy as np
import os

task = 'SA' # 'SA' or 'RC'

# Load the existing data files for ET and EEG
et_train_data = torch.load(f'./ET/{task}/kfold_train_raw_data.pt')
et_test_data = torch.load(f'./ET/{task}/kfold_test_raw_data.pt')

eeg_train_data = torch.load(f'./EEG/{task}/kfold_train_raw_data.pt')
eeg_test_data = torch.load(f'./EEG/{task}/kfold_test_raw_data.pt')

# Concatenate the "cognition" values from ET and EEG for train,and test data
train_data = []
test_data = []

print('Now getting the concatenated cognition values for train, and test data...')
for et_data, eeg_data in zip(et_train_data, eeg_train_data):
    et_cognition = et_data["cognition"]
    eeg_cognition = eeg_data["cognition"]
    if eeg_data["idx"] == 10:
        print(et_cognition.shape)
        print(eeg_cognition.shape)
    concatenated_cognition = np.concatenate((et_cognition, eeg_cognition), axis=1) # TODO: concatenate per sentence, not per word!!!
    
    train_data.append({
        "idx": eeg_data["idx"],
        "sentence": eeg_data["sentence"],
        "cognition": concatenated_cognition,
        "label": eeg_data["label"]
    })

for et_data, eeg_data in zip(et_test_data, eeg_test_data):
    et_cognition = et_data["cognition"]
    eeg_cognition = eeg_data["cognition"]
    concatenated_cognition = np.concatenate((et_cognition, eeg_cognition), axis=1)
    
    test_data.append({
        "idx": eeg_data["idx"],
        "sentence": eeg_data["sentence"],
        "cognition": concatenated_cognition,
        "label": eeg_data["label"]
    })

# Save the new train, dev, and test data as separate files
save_path = f'./ET+EEG/{task}'
if not os.path.exists(save_path):
    os.makedirs(save_path)
torch.save(train_data, f'{save_path}/kfold_train_raw_data.pt')
torch.save(test_data, f'{save_path}/kfold_test_raw_data.pt')

print('All Done!')