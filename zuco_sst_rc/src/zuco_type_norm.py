"""

Original file is located at
    https://colab.research.google.com/drive/1FOZLJhCO0DRtIsxWBhrW45i-tKVeT1iV

Excerpted from the third part for data extraction from Zuco, preparing and displaying the type aggregation data.

-YW

---
"""

#@title Type data Definitions
# https://github.com/DS3Lab/zuco-nlp/blob/master/sentiment-analysis/SST_experiment/stts_utils.py

# Setup


import os
import pandas as pd
import numpy as np
import random
import torch
import json
import math
import sys
import re
import pandas as pd
import time


"""---
# ZuCo

Import and explore, manipulate ZuCo type lexicons.

"""

sst_path = f"../data/"



def get_norm_vals(array):
    norm_vals = {}
    norm_vals["means"] = array.mean(axis = 0)
    norm_vals["stdevs"] = array.std(axis=0)
    norm_vals["stdevs"] = np.where(array.std(axis=0) > 0.0001, array.std(axis=0), 0.0001)
    return norm_vals

def normalize_vec(vec, norm_vals):
    return (vec - norm_vals["means"])/norm_vals["stdevs"]

def minmax(vec):
    vec = np.array(vec) if type(vec) == list else vec
    return (vec - vec.min())/(vec.max() - vec.min())

def maxx(vec):
    vec = np.array(vec) if type(vec) == list else vec
    return vec/vec.max()

def avg_electrodes(eeg_dict):
    return {word : np.array(eeg_dict[word]).mean()
                        for word in eeg_dict.keys()}

def max_electrodes(eeg_dict):
    return {word : np.array(eeg_dict[word]).max()
                        for word in eeg_dict.keys()}

def sum_electrodes(eeg_dict):
    return {word : np.array(eeg_dict[word]).sum()
                        for word in eeg_dict.keys()}

def normalize_embeddings(embeddings_dict, norm):
    all_embs = np.array([embeddings_dict[word] for word in embeddings_dict.keys()])
    if norm == "mean":
        norm_vals = get_norm_vals(all_embs)
        embeddings_dict = {word : normalize_vec(embeddings_dict[word], norm_vals)
                        for word in embeddings_dict.keys()}
    elif norm == "minmax":
        embeddings_dict = {word : minmax(embeddings_dict[word])
                        for word in embeddings_dict.keys()}
    elif norm == "max":
        embeddings_dict = {word : maxx(embeddings_dict[word])
                        for word in embeddings_dict.keys()}
    return embeddings_dict

def load_word_eeg(eeg_dictionary, word, dummy_eeg=np.zeros(105)):
    eeg_emb = eeg_dictionary.get(word, dummy_eeg)
    return eeg_emb

"""

#### Display type-aggregated data
"""
#@title Display ZuCo datasets (from type-aggregated datasets of RC - YW)
def get_zuco_d(norml=True,
                reduction = None, # 'avg', 'max', 'sum', None
                norm= 'mean', # ("Min-Max", "minmax"),("Z", "mean"),("Max", "max")] default: Z standardization - 'mean'
                mode :str = 'gaze', # 'eeg', 'gaze'
                                # 1. to decide pth
                                    # ("eeg", f"{sst_path}type_dict_eeg.json"),
                                    # ("gaze", f"{sst_path}type_dict_gaze.json"),],
                                # 2. to decide n values:
                                    # ("eeg", 105)
                                    # ("gaze", 5)
                # q = '' # query string
                ):
    if mode == "eeg":
        n = 104
        pth = f"{sst_path}type_dict_eeg.json"
    else:
        n = 5
        pth = f"{sst_path}type_dict_gaze.json"

    
    with open(pth, "r") as f:
        eeg_dictionary = json.load(f)
        # normalize
        if norml:
            eeg_dictionary = normalize_embeddings(eeg_dictionary, norm)
    # reduction
    if "eeg" in pth and not reduction:
        print(f"\nEEG\n")
        ddf = pd.DataFrame.from_dict(eeg_dictionary, orient='index').iloc[:, :n]
        ddf = ddf.reset_index()
        ddf.columns = ["token"] + [f"Electrode {i}" for i in range(1, n+1)]
    elif "eeg" in pth and reduction=='avg':
        eeg_dictionary = avg_electrodes(eeg_dictionary)
        ddf = pd.DataFrame.from_dict(eeg_dictionary, orient='index')
        ddf = ddf.reset_index()
        ddf.columns = ["token"] + ["Mean"]
    elif "eeg" in pth and reduction=='max':
        eeg_dictionary = max_electrodes(eeg_dictionary)
        ddf = pd.DataFrame.from_dict(eeg_dictionary, orient='index')
        ddf = ddf.reset_index()
        ddf.columns = ["token"] + ["Max"]
    elif "eeg" in pth and reduction=='sum':
        eeg_dictionary = sum_electrodes(eeg_dictionary)
        ddf = pd.DataFrame.from_dict(eeg_dictionary, orient='index')
        ddf = ddf.reset_index()
        ddf.columns = ["token"] + ["Sum"]
    else:
        # TODO: no reduction for gaze
        print("\nGaze\n")
        ddf = pd.DataFrame.from_dict(eeg_dictionary, orient='index')
        ddf = ddf.reset_index()
        ddf.columns = ["token", "Number of Fixations", "First Fixation Duration", "Gaze Duration",
                       "Go-Past Time", "Total Reading Time"]
    # if q and any((ddf.token==q).to_list()):
    #     print("q:", q)
    #     ddf = ddf[ddf.token==q]
    # display(ddf) # only available in Jupyter Notebook
    return ddf


# We use Z standardization norm for both EEG and gaze (Z is a must for gaze bcs of nfix) - YW
zuco_dictionary_eeg = get_zuco_d(norml=True,reduction = None, norm= 'mean', mode = 'eeg')
# print(zuco_dictionary_eeg)
zuco_dictionary_gaze = get_zuco_d(norml=True,reduction = None, norm= 'mean', mode = 'gaze')
print(zuco_dictionary_gaze.shape)

# Save zuco_dictionary_eeg and _gaze as a CSV file
zuco_dictionary_eeg.to_csv('../results/type_dictionary_eeg_normed.csv', index=True)
zuco_dictionary_gaze.to_csv('../results/type_dictionary_gaze_normed.csv', index=True)