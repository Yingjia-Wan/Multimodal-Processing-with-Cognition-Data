'''

Code by YINGJIA WAN, May 2023.
Code Goal: formatting the training data for sentiment anaysis from ZUCO, containing 400 sentences with labels (POSITIVE:2, NEUTRUAL: 1, NEGATIVE:0).

Input file: SA_data.csv, cognition_data/Sentence_data_ZAB.pickle
Output file: train_raw_data.pt, test_raw_data.pt


Some IMPORTANT data preprocessing tips:
1. In ZAB_SA_DATA SENTENCES，some errors are corrected:
    emp11111ty is corrected to empty in line 4;
2. However, puntucations are kept in order to align with the avg_eeg_sentences_zdn.tsv file.

'''
import os
import csv
import pandas as pd
import json
import pickle
import numpy as np
import sys
import torch
from sklearn.model_selection import train_test_split

task = 'SA' # 'SA' or 'RC'
# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Set a larger field size limit
csv.field_size_limit(sys.maxsize)

################################################## Create the new 'ZAB_SA_data.csv' corresponding to the sentence order in ZAB.pickle #################################################
# # Load pickle file
# with open('cognition_data/Sentence_data_ZAB.pickle', 'rb') as f:
#     cognition_data = pickle.load(f)

# # Write the data to a CSV file
# with open('ZAB_SA_data.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(['idx', 'sentence', 'label'])
#     for i in range(400):
#         idx = i
#         sentence = cognition_data[i]['content']
#         sentiment = cognition_data[i]['label_name']
#         if sentiment == 'NEGATIVE':
#             label = 'negative'
#         elif sentiment == 'NEUTRAL':
#             label = 'neutral'
#         elif sentiment == 'POSITIVE':
#             label = 'positive'
#         writer.writerow([idx, sentence, label])


################################################## Get eeg data dictionary #################################################


# def softmax(x):
#     exp_vals = np.exp(x)
#     sum_exp_vals = np.sum(exp_vals)
#     softmax_probs = exp_vals / sum_exp_vals
#     return softmax_probs

# Load the cognition data from the file
with open(f'./cognition_data/{task}/norm_avg_eeg.tsv', 'r') as file:
    eeg = csv.reader(file, delimiter='\t')
    header_row = next(eeg) # Skip the header row
    cognition_data = {}
    all_values = []
    eeg_rows = list(eeg)

    # iterate over each row in the input file, representing each word
    for row in eeg_rows:
        # eegvals
        cognition_string = row[3]
        cognition_string = cognition_string.strip("[]").split()
        cognition_embeds = [float(value) for value in cognition_string]
        cognition_embeds = np.array(cognition_embeds) # word-level eeg array
        # sentence_id
        sentence_id = row[0] # a string!
        # word_str
        word_str = row[2]

        # Create the cognition_data dictionary
        if sentence_id not in cognition_data:
            cognition_data[sentence_id] = {'words': [], 'embeds': []} 
        cognition_data[sentence_id]['words'].append(word_str)
        cognition_data[sentence_id]['embeds'].append(cognition_embeds)

        # Comment: Below can't handle repeated words in a sentence.
        # cognition_data[sentence_id][word_str] = cognition_embeds  # {sentence_id (str): 
                                                                            # {word_id (str): 
                                                                            #       {word_str (str): 
                                                                            #           cognition embeds (np array)}}


# Example: Access the first sentence and its words and embeds
first_sentence_id = list(cognition_data.keys())[0]
first_sentence_words = cognition_data[first_sentence_id]['words']
first_sentence_embeds = cognition_data[first_sentence_id]['embeds']
print(f'first_sentence_words: {first_sentence_words}')
print(f'Number of words in the first sentence: {len(first_sentence_words)}')
print(f'length of first_sentence_embeds: {len(first_sentence_embeds)}')


################################################## Save the train and test data #################################################
print('--------------------------------------------------')

# use the same splitting indices:
with open('kfold_splitting_indices.pkl', 'rb') as file:
    indices = pickle.load(file)
train_indices = indices["train_indices"]
test_indices = indices["test_indices"]

# Open the SA_data.csv file (SST text)
with open('ZAB_SA_data.csv', 'r') as infile:
    reader = csv.reader(infile)
    header_row = next(reader) # Skip the header row in the reader

    # Define the lists for train and test data
    train_data = []
    test_data = []

    eegvals = []

    # Iterate over each row in the input file
    for i, row in enumerate(reader):
        # 1. Text and label data:

        #       (1) Using the sentence data from the SST text file: ZAB_SA_data.csv
        # # Split the text sentence in the second column into words
        # exclude_chars = ["--", "...", "****.", "-", "&", "***."]
        # words = [word for word in row[1].split() if word not in exclude_chars]
        # # join words (a list variable) into sentence: 
        # sentence = ' '.join(words)
        # # sentence = row[1].replace(";", "")
        # # words = sentence.split()

        #       (2) Using the joined word data from the cognition_data dictionary
        sentence_id = row[0] # string
        words = cognition_data[sentence_id]['words'] # list
        sentence = ' '.join(words)
        
        # 2. EEG data
        # convert the nested dictionary of cognition_data[sentence_id] to a list of word-level eegvals.
        sentence_eegvals = []
        for j, word in enumerate(words):
            eegvals = cognition_data[sentence_id]['embeds'][j]
            sentence_eegvals.append(eegvals) # TODO: the word-level eegvals are not separated by a special token!
        if i == 0:
            print('The first sentence to be appended to datasets:')
            print(f'type(sentence_eegvals): {type(sentence_eegvals)}')
            print(f'its np array shape: {np.array(sentence_eegvals).shape}')
            print(f'sentence: {sentence}') # TODO: double check if the sentence is correct and COMPLETE!
            print(f'label: {row[2]}')
            print(f'idx: {i}')

        # # Randomly assign the row to the train or test set
        if i in train_indices:
            train_data.append({
                "idx": i,
                "sentence": sentence, #row[1] Note that the modified data contains preprocessed sentence (trimed off some punctuations), in order to aligh sequence length with number of cognition embeddings.
                "cognition": np.array(sentence_eegvals), # TODO: eegvals betwee words are not separated by a special token!!!!
                "label": row[2]
            })
        elif i in test_indices:
            test_data.append({
                "idx": i,
                "sentence": sentence,
                "cognition": np.array(sentence_eegvals),
                "label": row[2]
            }) # train/test_indices are lists

# Example: sentence idx = 12, positive, test
print('--------------------------------------------------')
print('Example: sentence idx = 0, neutral, from test')
print(f'The first sentence (checked from splitting indices its in test_data): {test_data[0]}')
print(f'Shape of the EEG data of the sentence: {test_data[0]["cognition"].shape}')
print(f'The EEG data of the sentence: {test_data[0]["cognition"]}')


save_path = f'./EEG/{task}'
if not os.path.exists(save_path):
    os.makedirs(save_path)
# Save the train, and test data as separate files
torch.save(train_data, f'{save_path}/kfold_train_raw_data.pt')
torch.save(test_data, f'{save_path}/kfold_test_raw_data.pt')
print(f'Successfully saved the train, and test EEG data for the {task} task!', '\n')