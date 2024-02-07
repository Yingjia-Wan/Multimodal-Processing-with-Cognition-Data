# Setup
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

# Data

## A. Data source

After preproessing, the preprocessed .tsv data were saved in `zuco_preprocessing/results`. You can copy the data file to the `data/cognition_data` folder in the current directory for convenience.


## B. Data processing instructions for training
0. set splitting ratio and random seed for splitting in splitting_indices.py
1. run splitting _indices.py
    - if k-fold cross validation, only split the dataset into train (later containig both train and dev) and test set.
2. run three data_grouping.py to generate train, val and test set of ET, EEG, and ET+EEG data.

## C. Data format
The data_grouping.py will generate .pt data for training. For each sample it contains:

{
        
        "idx": int of the sentence index in the original dataset,
        
        "sentence": sentence string joined from the words from the norm_avg_data.tsv,
        
        "cognition": np array of shape (n, 5/104/109) where n is the number of words in the sentence.

        "label": str, negative/positive/neutral
    }


# CogMAPL Architecture

The CogMAPL structure is specified in `COGMAPL_decoder.py`. Miscellaneous functions are in `utils.py` and `dataset.py`. 

In a nutshell, the COGMAPL model consists of several components:

    1. A special tokenizer:
    adjusted to be able to accept both cognition data and textual data.

    2. A Projection Layer: 
    added so that the cognition data can be projected to the same dimension as the word embeddings of the baseLM.

    3. baseLM: 
    The decoder-based LM (e.g., GPT-2) is added with a classfication head to perform the sentiment analysis task.


[Acknowledgements]:

- A part of the architecture code takes reference from the Github repo:
https://github.com/mair-lab/mapl/blob/main/mapl.py about the MAPL framework built by Oscar Mañas, et al., 2023.

- GPT2ForSequenceClassification is from Huggingface:
https://huggingface.co/transformers/v4.8.0/_modules/transformers/models/gpt2/modeling_gpt2.html#GPT2ForSequenceClassification
https://huggingface.co/transformers/v4.8.0/model_doc/gpt2.html#gpt2forsequenceclassification


# Training & Evaluation

To train the model, run 'train.py' in the CogMAPL folder.

The following hyperparameters should be specified:

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