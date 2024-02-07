import os
import csv
import pandas as pd
import json
import pickle
import numpy as np
import sys
import torch


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


################################################## Get et data dictionary #################################################


# def softmax(x):
#     exp_vals = np.exp(x)
#     sum_exp_vals = np.sum(exp_vals)
#     softmax_probs = exp_vals / sum_exp_vals
#     return softmax_probs

# Load the cognition data from the file
with open(f'./cognition_data/{task}/norm_avg_gaze.tsv', 'r') as file:
    gaze = csv.reader(file, delimiter='\t')
    header_row = next(gaze) # Skip the header row
    cognition_data = {}
    all_values = []
    gaze_rows = list(gaze)

    # iterate over each row in the input file, representing each word
    for row in gaze_rows:
        # etvals
        etvals = []
        FFD = float(row[3])
        GD = float(row[4])
        GPT = float(row[5])
        TRT = float(row[6])
        nFix = float(row[7])
        etvals.append(FFD)
        etvals.append(GD)
        etvals.append(GPT)
        etvals.append(TRT)
        etvals.append(nFix)
        etvals = np.array(etvals)
        # sentence_id
        sentence_id = row[0] # a string!
        # word_str
        word_str = row[2]

        # Create the cognition_data dictionary
        if sentence_id not in cognition_data:
            cognition_data[sentence_id] = {'words': [], 'embeds': []}
        cognition_data[sentence_id]['words'].append(word_str)
        cognition_data[sentence_id]['embeds'].append(etvals)
        #   cognition_data[sentence_id] = {'words': [], 'FFD': [], 'GD': [], 'GPT': [], 'TRT': [], 'nFix': []}


# Example: Access the first sentence and its words and embeds
first_sentence_id = list(cognition_data.keys())[0]
first_sentence_words = cognition_data[first_sentence_id]['words']
first_sentence_embeds = cognition_data[first_sentence_id]['embeds']
print(f'first_sentence_words: {first_sentence_words}')
print(f'Number of words in the first sentence: {len(first_sentence_words)}')
print(f'shape of first_sentence_embeds: {len(first_sentence_embeds)}')

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

    etvals = []
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
        
        # 2. Gaze data
        # convert the nested dictionary of cognition_data[sentence_id] to a list of word-level etvals.
        sentence_etvals = []
        for j, word in enumerate(words):
            etvals = cognition_data[sentence_id]['embeds'][j]
            sentence_etvals.append(etvals) # TODO: important! the ET data is not separated by a special token!
        if i == 0:
            print('The first sentence to be appended to datasets:')
            print(f'type(sentence_etvals): {type(sentence_etvals)}')
            print(f'its np array shape: {np.array(sentence_etvals).shape}')
            print(f'sentence: {sentence}') # TODO: double check if the sentence is correct and COMPLETE!
            print(f'label: {row[2]}')
            print(f'idx: {i}')

        # # Randomly assign the row to the train, dev, or test set
        if i in train_indices:
            train_data.append({
                "idx": i,
                "sentence": sentence, #row[1] Note that the modified data contains preprocessed sentence (trimed off some punctuations), in order to aligh sequence length with number of cognition embeddings.
                "cognition": np.array(sentence_etvals), # TODO: etvals betwee words are not separated by a special token!
                "label": row[2]
            })
        elif i in test_indices:
            test_data.append({
                "idx": i,
                "sentence": sentence,
                "cognition": np.array(sentence_etvals),
                "label": row[2]
            }) # train/test_indices are lists

# Example: sentence idx = 12, positive, test
print('--------------------------------------------------')
print('Example: sentence idx = 0, neutral, from test')
print(f'The first item of test (you can check from splitting indices which sentence): {test_data[0]}')


save_path = f'./ET/{task}'
if not os.path.exists(save_path):
    os.makedirs(save_path)
# Save the train, and test data as separate files
torch.save(train_data, f'{save_path}/kfold_train_raw_data.pt')
torch.save(test_data, f'{save_path}/kfold_test_raw_data.pt')
print(f'Successfully saved the train, and test ET data for the {task} task!')