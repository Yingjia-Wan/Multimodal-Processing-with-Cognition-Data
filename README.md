# Multimodal-Processing-with-Cognition-Data
Implementation code for my Master's Dissertation: "Multimodal Prompt-Tuning with Human Cognition Data" by Yingjia Wan.

# Data Source
- The original ZuCo corpus is available at https://osf.io/uxamg/. As the size of the data files is huge (over 50 GB), it is recommended to download them efficiently in batch using the code in `./zuco_preprocessing/src/zuco_matfiles_download.py`, rather than downloading manually from the website. See systematic instructions below.

- The `zuco_data_storage` folder in the current dir is imported from [zuco-nlp](https://github.com/DS3Lab/zuco-nlp/tree/master) for formatting and extracting from neuro signals for NLP research. See [zuco_data_storage/README.md](./zuco_data_storage/README.md) for details.

# CogMAP

## Setup
```
conda create -n mapl python=3.8
conda activate mapl

#install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install scikit-learn
conda install jupyter
(you may also need to: conda update jupyter ipywidgets)

# install other dependencies
pip install -r requirements.txt

# Install helper functions for training
!pip install -q git+https://github.com/gmihaila/ml_things.git

```

## Data

### A. Preprocessing (i.e., extracting cognition features from raw neural signals)
First, please refer to the [README.md](./zuco_preprocessing/README.md) in the folder `zuco_preprocessing` for data formatting and preprocessing instructions.

After preproessing, the preprocessed .tsv data were saved in `zuco_preprocessing/results`. You can move the data file to the `data/cognition_data` folder in the current directory for convenience.

## B. Preparing data for training
1. run `splitting _indices.py`
    - set splitting ratio and random seed
    - if k-fold cross validation, the dataset will be only split into train (later containig both train and dev) and test set.
3. run `data_grouping.py` to generate train, val and test set of ET, EEG, and ET+EEG data.

## C. Data format
The data_grouping.py will generate .pt data for training. For each sample it contains:

  `"idx"`: int of the sentence index in the original dataset,
  
  `"sentence"`: sentence string joined from the words from the norm_avg_data.tsv,
  
  `"cognition"`: np array of shape (n, 5/104/109) where n is the number of words in the sentence.

  `"label"`: str, negative/positive/neutral


## CogMAPL Model Architecture

The CogMAPL architecture is specified in `COGMAPL_decoder.py`. Miscellaneous functions are in `utils.py` and `dataset.py`. 

In a nutshell, the COGMAPL model consists of three key components:

1. A special tokenizer:
adjusted to be able to accept both cognition data and textual data.

2. A Projection Layer: 
added so that the cognition data can be projected to the same dimension as the word embeddings of the baseLM. The layer takes reference from the [MAPL](https://github.com/mair-lab/mapl/blob/main/mapl.py) framework.

3. base LM: 
The decoder-based LM (e.g., GPT-2) is added with a classfication head to perform the sentiment analysis task.(e.g., [GPT2ForSequenceClassification](https://huggingface.co/transformers/v4.8.0/model_doc/gpt2.html#gpt2forsequenceclassification))


## Training & Evaluation

To train the model, run `train.py` in the CogMAPL folder.

The following hyperparameters need to be specified:

```
seeds=(16 17 18 19 20)
seed=${seeds[N]}
# max_sequence_len=1024
epoch=60
cognition_type='ET'
model='gpt2-medium'
lr=3e-5
batch_size=16

python training.py\
    --model_name_or_path $model \
    --cognition_type $cognition_type \
    --batch_size $batch_size \
    --learning_rate $lr \
    --num_epochs $epoch \
    --seed $seed \
```


