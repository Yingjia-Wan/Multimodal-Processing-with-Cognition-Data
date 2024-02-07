## Acknowledgement

This code was adapted from the following paper: 

Nora Hollenstein, Maria Barrett, Marius Troendle, Francesco Bigiolli, Nicolas Langer & Ce Zhang. _Advancing NLP with Cognitive Language Processing Data_. 2019.
https://arxiv.org/abs/1904.02682

## Data

You will need the following data: (They can be efficiently downloaded via zuco_sst_rc dir.) \
	- The ZuCo data in their latest (January 2019) format. Those data must be placed in the `Data_to_preprocess/` folder. The data can be accesed here: https://osf.io/uxamg/


## Setp up and Code

conda activate zuco

1. You will need an python=3.8 environment with the packages in requirements.txt

2. After setting up the environment and all the data, extract them (.mat) to (.pkl) for each subject. To do that, run `python create_modeling_data.py` from this directory. 

Potential paramenters for this run are `-s` if you want to save a report of this preprocessing and `-low_def` if you want to save the newly preprocessed EEG signal (the most intensive component memory-wise) with low definition (np.float16).
 `python create_modeling_data.py -low_def -s`

## Changes made in zuco_nlp/sentiment-analysis:

1. added requirements.txt
2. constants.py: 
	- FEATURE_DIMENSIONS = {"RAW_ET":5,
                      "RAW_EEG":105,
                      "ICA_EEG":61
                      } 

	(seems like it doesn't matter in the original code, what may matter is in tf_modeling.py: self.et_features_size = 5)
	
	[Update]: It is later found that the EEG raw data in ZUCO matfiles only has 104 eeg values, with the 105th being 0s! However we don't need to change the code here. Simply keep it this way at 105 values. Instead the valid 104 features are extracted later in zuco_sst_rc.
	- SUBJECT_NAMES: (1) removed ZDN because its matfile is corrupted; (2) changed ZMG to ZMB according to zuco osf.

3. added Data_to_preprocess folder: download matlab files, using code in the folder 'zuco_sst_rc/src':
4. added log and Results_files folder to save outputs (pkl files).
5. changes in create_modeling_data.py: filepaths
6. changes in data_loading_helper.py:
	- changed N_ET_VARIABLES = 5 (seems like it doesn't matter in the original code)
	- changed accordingly: def compress_eye_tracking_by_word
	- replaced all 'new_config' with 'config' (also in tf_modeling.py)
	- removed tflearn and vocab_processor, replaced with the tokenizer and pad_sequence from tensorflow.keras.preprocessing (see in the code: def get_processed_dataset)
	- [PEDNING]: (pre?)normalization method: 
7. changes in data_creation_helper.py:
	- def extract_sentence_level_data


## using slum:
1. submit a job:
`sbatch xxx.sh`
check queue:
`squeue`
2. switching GPU:
`#SBATCH --gres=gpu:rtx_6000_ada:1`
`#SBATCH --gres=gpu:rtx_4090:1`
3. cancel/withdraw a job:
`scancel JOBID`