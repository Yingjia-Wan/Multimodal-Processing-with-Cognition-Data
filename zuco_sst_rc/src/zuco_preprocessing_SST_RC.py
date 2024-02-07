"""

Original file is located at
    https://colab.research.google.com/drive/1FOZLJhCO0DRtIsxWBhrW45i-tKVeT1iV

Excerpted from the third part for data extraction from Zuco, which include:
*   (3a) preparing and displaying the type aggregation data
*   (3b) preparing the piece contextualized data.
We used the TreeBankTokenizer for displaying the 105 electronodes and 5 eyetracking features (so purely on word-level instead of token-level).

-YW

---
"""

"""---
# ZuCo

Create ZuCo datasets, examine coverage, create piece type lexicons

"""


#### -1. SETUP


# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', additional_special_tokens=["<e>", "</e>"])
# ztokenizer = TreebankWordTokenizer()
# detokenizer = TreebankWordDetokenizer(
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize.treebank import TreebankWordTokenizer
from zipfile import ZipFile
import pickle as pkl
import pandas as pd
import numpy as np
import scipy.io
import pprint
import h5py
import ast
import sys
import os
import random
import re
import json
from zuco_utils import *
# from zuco_dataset import *
from scipy.io import loadmat, savemat
import pprint
# import yaml


ZNLP = "../../zuco-nlp"
ztokenizer = TreebankWordTokenizer()
#texts = [" ".join(ztokenizer.tokenize(s)) for s in texts]
task = 'SA' # switch between 'SA' and 'RC'.

if task == 'SA':
    SUBJECT_NAMES = ["ZAB", "ZDM", 
                    #  "ZDN", # corrupted, abandoned - YW
                    "ZGW", "ZJM", "ZJN", "ZJS", "ZKB", "ZKH", "ZKW", "ZMB", "ZPH"] # task1, 11 subjects
if task == 'RC':
    SUBJECT_NAMES = ["ZPH", "ZMG", "ZAB", "ZJN", "ZKH", "ZGW", "ZKB", "ZDM","ZDN", "ZJM", "ZKW"] # task2-normal reading, 11 subjects, lacking ZJS


NR_OLD = ["ZPH", "ZJS"]
SR_OLD = ["ZDN"] # Abandoned - YW

FEATURE_DIMENSIONS = {"RAW_ET":5,
                      "RAW_EEG":104,
                      "ICA_EEG":61
                      }

ANSWERED_SENTENCES_INDICES = np.array([3, 11, 12, 16, 21, 25, 28, 38, 49, 55, 68, 78, 88,
                                       103, 122, 124, 128, 135, 137, 152, 156, 161, 171, 176, 183, 184,
                                       187, 190, 193, 201, 215, 227, 262, 265, 266, 274, 287, 318, 363,
                                       374, 375, 383, 391, 394, 395, 397])

EEG_SENTIMENT_ELECTRODES = [2,3,4,5,7,8,9]

stopwords = {"an", "for", "do", "its", "of","off", "is", "s", "am", "or", "as",
             "from", "him", "each", "the", "are", "we", "these", "his", "me",
             "were", "her", "this", "our", "their", "up", "to", "ours", "she",
             "at", "them", "and", "in", "on", "that", "so", "did", "now", "he",
             "you", "has", "i", "t", "my", "a", "by", "it"}


#### 0. MATFILES

# Assuming: Downloaded the matfiles from the ZUCO website and converted them to pkl files. code in 'zuco-nlp'.
if task == 'SA':
    mat_path = f"{ZNLP}/sentiment-analysis/Data_to_preprocess"
    pkl_path = f"{ZNLP}/sentiment-analysis/Results_files"
if task == 'RC':
    mat_path = f"{ZNLP}/relation-classification/Data_to_preprocess"
    pkl_path = f"{ZNLP}/relation-classification/Results_files"

tsv_path = f"../results/subject_tsv/{task}"
if not os.path.exists(tsv_path):
    os.makedirs(tsv_path)

eeg_tsv_path = os.path.join(tsv_path, "eeg_tsv")
if not os.path.exists(eeg_tsv_path):
    os.makedirs(eeg_tsv_path)

gaze_tsv_path = os.path.join(tsv_path, "gaze_tsv")
if not os.path.exists(gaze_tsv_path):
    os.makedirs(gaze_tsv_path)



'''

#### 1. Convert subject pickles to dfs, then to .tsv (save per subject in SUBJECT_NAMES) 
# # see pkl file for details if confused. - YW

"""EEG tsv, save per subject."""

for subject in SUBJECT_NAMES:
    pkl_df = pd.read_pickle(f"{pkl_path}/Sentence_data_{subject}.pickle")
    df = pd.DataFrame()
    print(f'Extracting: {subject} EEG.')
    
    word_access_error = 0
    eeg_access_error = 0

    # sentence-level
    for j in range(len(pkl_df.keys())):
        sentence_length = len(pkl_df[j]['word_level_data'].keys())
        sentence_content = pkl_df[j]['content']

        # word-level
        for i in range(sentence_length-1):

            # Error 1: word-level 'content' not accessible. # KeyError
            try:
                word = pkl_df[j]['word_level_data'][i+1]['content']
            except KeyError as e:
                # print(f"KeyError: {e}")
                # print(f"Failed to get word-level 'content' from {subject}. word {i+1} in sentence {j}: {sentence_content}")
                # word = "N/A"  # or some default value
                word_access_error += 1
                if j == 40:
                    word = "uplifter."
                elif j == 41:
                    word = "journey."
                elif j == 353:
                    word = "incredible!"
                else:
                    print(f"New error. Failed to get word-level 'content' from {subject}. word {i+1} in sentence {j}: {sentence_content}")
                    word = "N/A"

            # Error 2: EEG values not accessible. # KeyError
            try:
                eegvalues = pkl_df[j]['word_level_data'][i+1]['RAW_EEG'][0][:, :104].mean(axis=0, keepdims=True)
            except KeyError as e:
                # if subject is 'ZGW': # there is a keyerror [3] in ZGW
                #     print(f'keyerror: {e}')
                #     print(f"sentence {j}: Null EEG values for {word} in: {sentence_content}")
                eegvalues = np.zeros((1, 104))
                eeg_access_error += 1

            res = pd.DataFrame.from_dict({'sent': j,
                                          'wordix': i,
                                          'word': word,
                                          'eegvals': eegvalues},  # 104 columns
                                         orient='index').T
            df = pd.concat([df, res])

    print(f'{subject} word_access_error: {word_access_error}.')
    print(f'{subject} eeg_access_error: {eeg_access_error}.', '\n')
    df.to_csv(f"{eeg_tsv_path}/{subject}.tsv", sep="\t")


"""ET tsv, save per subject."""

for subject in SUBJECT_NAMES:
    pkl_df = pd.read_pickle(f"{pkl_path}/Sentence_data_{subject}.pickle")
    df = pd.DataFrame()
    print(f'Extracting: {subject} ET.')

    word_access_error = 0
    et_null_error = 0
    et_access_error = 0

    # sentence-level
    for j in range(len(pkl_df.keys())):
        sentence_length = len(pkl_df[j]['word_level_data'].keys())
        sentence_content = pkl_df[j]['content']
        # print(f'sentence {j+1}: {pkl_df[j]["content"]}')
        # print(f'word-level Keys: ', pkl_df[j]['word_level_data'].keys()) # word-level Keys:  dict_keys([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 'word_reading_order'])
        # print(f'Number of words in sentence {j+1}: {sentence_length}.') # 23 = 22 (words) + 'word_reading_order'
        
        # word-level
        for i in range(sentence_length-1):

            # Error 1: word-level 'content' not accessible. # KeyError
            try:
                word = pkl_df[j]['word_level_data'][i+1]['content']
            except KeyError as e:
                # print(f"KeyError: {e}")
                # print(f"Failed to get word-level 'content' from {subject}. word {i+1} in sentence {j}: {sentence_content}")
                # word = "N/A"
                if j == 40:
                    word = "uplifter."
                elif j == 41:
                    word = "journey."
                elif j == 353:
                    word = "incredible!"
                else:
                    print(f"New error. Failed to get word-level 'content' from {subject}. word {i+1} in sentence {j}: {sentence_content}")
                    word = "N/A"
                word_access_error += 1

            # # IMPORTANT: We use the already preprocessed word-level ET feature values, instead of raw ET values. -YW
            # etvalues = pkl_df[j]['word_level_data'][i+1]['RAW_ET'][0].mean(axis=0,keepdims=True) # Discarded!

            try:
                # Error 2: ET values accessible, but is None.  # AttributeError
                try:
                    TRT = pkl_df[j]['word_level_data'][i+1]['TRT'].mean() # TRT encompasses all eye movements, so it ensure whether the word has null ET values. -YW
                    FFD = pkl_df[j]['word_level_data'][i+1]['FFD'].mean() #.mean() doesn't have an effect, just making sure each feature value is a scalar.
                    GD = pkl_df[j]['word_level_data'][i+1]['GD'].mean()
                    GPT = pkl_df[j]['word_level_data'][i+1]['GPT'].mean()
                    nFix = pkl_df[j]['word_level_data'][i+1]['nFix'].mean()
                except AttributeError:
                    # print(f"sentence {j}: Null ET values for {word} in: {sentence_content}")
                    FFD = 0
                    GD = 0
                    GPT = 0
                    TRT = 0
                    nFix = 0
                    et_null_error += 1

            # Error 3: ET values not accessible. # KeyError
            except KeyError as e:
                # print(f"KeyError: {e}")
                # print(f"Failed to get word-level ET values from {subject}. word {i+1} in sentence {j}: {sentence_content}")
                FFD = 0
                GD = 0
                GPT = 0
                TRT = 0
                nFix = 0
                et_access_error += 1

            res = pd.DataFrame.from_dict({'sent': j,
                                            'wordix': i,
                                            'word': word,
                                            'FFD': FFD,
                                            'GD': GD,
                                            'GPT': GPT,
                                            'TRT': TRT,
                                            'nFix': nFix
                                            },
                                            orient='index').T
            df = pd.concat([df, res])

    print(f'{subject} word_access_error: {word_access_error}.')
    print(f'{subject} et_null_error: {et_null_error}.')
    print(f'{subject} et_access_error: {et_access_error}.', '\n')
    df.to_csv(f"{gaze_tsv_path}/{subject}.tsv", sep="\t")



# Error analysis: -YW 

# eeg_access_error = et_null_error. Thus we can infer that null ET is indeed the cause for EEG access error. 
# Because null ETs are normal expected events (i.e., no fixations), we only need to check Error 1: WORD-LEVEL CONTENT ACCESS ERROR.

# Summary of 3 word_access_error:
# KeyError: 3
# Failed to get word-level 'content' from ZGW. word 3 in sentence 40: Reassuring, retro uplifter.
# KeyError: 6
# Failed to get word-level 'content' from ZGW. word 6 in sentence 41: Flaccid drama and exasperatingly slow journey.
# KeyError: 4
# Failed to get word-level 'content' from ZJM. word 4 in sentence 353: Gollum's `performance' is incredible!

# Check the sentences in subject pkl file: via zuco-nlp/sentiment-analysis/Results_files/read_pickle.py
# Conclusion: Word_access errors are likely to be caused by all None ET values in a sentence. So it is perfectly fine we set them to 0.


print(f'Successfully saved subject tsv files for task {task} in {eeg_tsv_path} and {gaze_tsv_path}.', '\n')

'''

"""##### 2. Organize the tsv values to np, transforms tsv to df with the first null column dropped, and gather all subjects df in a list: dfs.
"""

def get_allsubs_dfs(mod:str=''):
    dfs = []
    if mod == "gaze":
        shape = (1, 5)  # corrected to 5 fixation features (e.g., FFD), from 4 raw ET features (time, area, x, y). - YW
    elif mod == "eeg":
        shape = (1, 104)

    for subject in SUBJECT_NAMES:
        if mod == "gaze":
            pth = f'{gaze_tsv_path}/{subject}.tsv'
        elif mod == "eeg":
            pth = f'{eeg_tsv_path}/{subject}.tsv'
        dfs.append(load_df(pth, mod))
    return dfs

# Collect subject dataframes per modality.
dfs_et = get_allsubs_dfs(mod = 'gaze')
dfs_eeg = get_allsubs_dfs(mod = 'eeg')

# print(f'Number of subjects in dfs_et: {len(dfs_et)}.')
# print(f'Number of subjects in dfs_eeg: {len(dfs_eeg)}.', '\n')

# print(f'Type of dfs_et: {type(dfs_et)}.')
# print(f'Type of dfs_et[0]: {type(dfs_et[0])}.', '\n')

# print('ZAB ET:','\n', dfs_et[0].head(3))
# print('ZAB EEG:','\n', dfs_eeg[0].head(3))

# print(f'eegvals of ZAB EEG df:', '\n', dfs_eeg[0].iloc[:, -1])
# print(f'Type of eegvals of ZAB EEG df:', '\n', type(dfs_eeg[0].iloc[:, -1])) # TODO: check the eegvals type. - YW
print(f"The number of columns in ZAB ET df is: {dfs_et[0].shape[1]}.")


##### 3. Extracting text data from ZUCO

"""
----RC
Input: textfiles from zuco-nlp/task/data
output: clean the sentences, extract entities and x, y sentences (no labels)

----SA
input: textfiles from zuco-nlp/task/data
Output: texts (no labels)
"""

# sorted(l, key=lambda i: int(i[:i.find("_")])))

if task == 'RC':
    cfg = read_config_file("/../../zuco-nlp/relation-classification/config.yml")
    dataset_path = f"{ZNLP}/relation-classification/data/zuco_nr_cleanphrases/"
    dataset, x_text, y, x_text_entities = get_processed_dataset_RC(dataset_path=dataset_path,
                                                                    binary=False, verbose=True,
                                                                    labels_from=None)
    sentence_order = get_sentence_order(dataset, task)
    x = np.array(x_text)
    entities = np.array(x_text_entities)
    '''
    :return:
    dataset: dict with keys: data, target, target_names, filenames, DESCR
    x: np.array of sentences
    y: np.array of label indexes
    entities: np.array of entities text with tags
    '''

elif task == 'SA':
    texts = [i for i in range(400)]
    # Get SST sentences
    with ZipFile(f"{ZNLP}/sentiment-analysis/data/all_sentiment_sentences.zip") as zf:
        for fname in zf.infolist():
            if 'all' in fname.filename:  # e.g., filename: all/NEGATIVE/2.txt
                if "txt" in fname.filename:
                    ix = int(fname.filename.split("/")[2].split(".")[0]) # after split: ["all", "NEGATIVE", "2.txt"],
                    # so ix would be 2.
                    with zf.open(fname) as f:
                        z = pd.read_csv(f,
                                        sep="\t",
                                        header=None,
                                        names=['text'])
                        text = z.text.values.tolist()
                        if text:
                            texts[ix] = text[0]


    dataset_path = f"{ZNLP}/sentiment-analysis/data/sentences"
    dataset, x_text, y = get_processed_dataset_SA(dataset_path, binary = False, verbose=True, 
                                                  labels_from='all') # see zuco-nlp/sentiment-analysis/data:
                                                                        # 'all' means all 400 sentences in the experiments, so output y is the correct labels.
                                                                        # None - subject_folder means 47 sentences in control conditions, so output y is the response answered by the subject.
    # Example: Print the first few sentences and their corresponding labels
    for i in range(3):
        print("Sentence:", x_text[i])
        print("Label:", y[i])
        print("---")
    x = np.array(x_text)
    sentence_order = get_sentence_order(dataset, task)

    '''
    :return:
    dataset: dict with keys: data, target, target_names, filenames, DESCR
    x: np.array of sentences
    y: np.array of label indexes
    '''





##### 4. Averaging over all subjects
"""
# Assuming you have already loaded the dataframes for EEG and gaze data for all subjects
# dfs_eeg and dfs_et are lists of dataframes for EEG and gaze, respectively.
"""

def average_subjects_data(dfs, columns_to_average):
    # Concatenate all dataframes into a single dataframe
    concatenated_df = pd.concat(dfs)
    # Group features by 'sent', 'wordix', and 'word' and calculate the mean for each group
    averaged_df = concatenated_df.groupby(['sent', 'wordix', 'word'])[columns_to_average].mean().reset_index()
    return averaged_df

# 5 separate ET features to average over subjects.
gaze_columns = ['FFD', 'GD', 'GPT', 'TRT', 'nFix']

# Average over subjects
averaged_gaze_df = average_subjects_data(dfs_et, gaze_columns)
average_eeg_df = average_subjects_data(dfs_eeg, ['eegvals'])
print(f"The number of columns in averaged_gaze_df is: {averaged_gaze_df.shape[1]}.") # should be 8
print(f"The number of columns in average_eeg_df is: {average_eeg_df.shape[1]}.") # should be 4



##### 5. Normalize the word-level feature values over their sentence.
""" 
i.e., extract the relative word importance within each sentence by normalizing the word-level feature values over their sentence.
"""

# innitialize
norm_avg_gaze_df = averaged_gaze_df.copy() # to preserve the original averaged_gaze_df
norm_avg_eeg_df = average_eeg_df.copy()

# z-score normalization/standardization

# (1) gaze: each gaze column cell contains a scalar.
norm_avg_gaze_df[gaze_columns] = norm_avg_gaze_df.groupby('sent')[gaze_columns].transform(lambda x: (x - x.mean()) / x.std())

# (2) eeg: each eegvals column cell contains a 1D nparray.
# print(f'The first word in the first sentence: {norm_avg_eeg_df["word"][0]}.') # Reassuring, 'Presents'
# print(f"Shape of eegvals of the first word: {norm_avg_eeg_df['eegvals'][0].shape}.") # (1, 104)

def normalize_eegvals(sent_group):
    # Initialize
    norm_group = sent_group.copy()

    # Compute mean and std across the 'eegvals' column
    mean_eeg = np.mean(np.vstack(norm_group['eegvals'].values), axis=0)
    std_eeg = np.std(np.vstack(norm_group['eegvals'].values), axis=0)

    # Normalize 'eegvals'
    norm_group['eegvals'] = norm_group['eegvals'].apply(lambda x: (x - mean_eeg) / std_eeg)
    return norm_group

norm_avg_eeg_df = norm_avg_eeg_df.groupby('sent').apply(normalize_eegvals)


""" 6. Saving average_xx_df to tsv files."""

save_path = f"../results/avg_task_specific/{task}"
if not os.path.exists(save_path):
    os.makedirs(save_path)

averaged_gaze_df.to_csv(f"{save_path}/avg_gaze.tsv", sep="\t", index=False)
average_eeg_df.to_csv(f"{save_path}/avg_eeg.tsv", sep="\t", index=False)
print(f'Successfully saved averaged tsv files for task {task} in {save_path}.', '\n')

norm_avg_gaze_df.to_csv(f"{save_path}/norm_avg_gaze.tsv", sep="\t", index=False)
norm_avg_eeg_df.to_csv(f"{save_path}/norm_avg_eeg.tsv", sep="\t", index=False)
print(f'Successfully saved normalized averaged tsv files for task {task} in {save_path}.', '\n')
