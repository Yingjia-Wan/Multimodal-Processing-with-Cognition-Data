## Acknowledgement

This folder was largely adapted from the following paper for data extraction: 

"Nora Hollenstein, Maria Barrett, Marius Troendle, Francesco Bigiolli, Nicolas Langer & Ce Zhang. _Advancing NLP with Cognitive Language Processing Data_. 2019.
https://arxiv.org/abs/1904.02682"

Please also cite the above paper if you use the code in this folder.

## Data

You will need the following data. 
- The ZuCo data in their latest (January 2019) format (https://osf.io/uxamg/).
	- The data must be placed in the `Data_to_preprocess/` folder.
	- They can be efficiently downloaded using the code from the paralell folder zuco_preprocessing by Yingjia Wan. Mannual download is not recommended due to the large size of the files.



## Setp up and Code

1. You will need an python=3.8 environment with the packages in requirements.txt

2. After setting up the environment and all the data, extract them (.mat) to (.pkl) for each subject. To do that, run `python create_modeling_data.py` from this directory. 

Potential paramenters for this run are `-s` if you want to save a report of this preprocessing and `-low_def` if you want to save the newly preprocessed EEG signal (the most intensive component memory-wise) with low definition (np.float16).
 `python create_modeling_data.py -low_def -s`

