# Acknowledgement
The zuco_preprocessing folder can be used to extract task-specific word-level gaze and EEG data from the ZUCO corpus. It preprocesses the .mat data to .tsv files by performing downloading, organizing on the word-level, converting file formats, etc.

- A part of this code refers to https://drive.google.com/drive/folders/1_zJRdPai1Y9NQkutPvqjUwb2Rs0qt6l1?usp=drive_link and https://colab.research.google.com/drive/1q6InysgFbFq5I2ULuBa91krkktvcJ4fe?usp=sharing by Erick Mcguire. However, extensive changes made as addressed below.

- The original ZuCo corpus is available at https://osf.io/uxamg/

- This project also requires a `zuco_data_storage` folder parallel to the current dir, which is downloaded from https://github.com/DS3Lab/zuco-nlp/tree/master. I rename the folder name `zuco-nlp` to `zuco_data_storage`.

# Instructions for data preprocessing:
### 1. Download:
Download matlab files (very large) from ZuCo by running code in the folder `zuco_preprocessing/src`: `zuco_matfiles_download.py`. The matfiles will be saved in the folder `zuco_data_storage/XXXXX/Data_to_preprocess`. (Mannual download is not recommended due to the large size of the files.)

### 2. Convert:
Convert the matfiles into pkl files, using code in the folder e.g, `zuco_data_storage/sentiment-analysis/`: `create_modeling_data.py`.
This will create the pkl files for each subject in the subfolder `../Result_files`

### 3. Merge:

Merge the subject-pkl files into csv files averaging over all subjects, create word-level gaze and EEG datasets, by running code in the folder `zuco_preprocessing/src/zuco_preprocessing_SST_RC.py`.
The result avg.tsv files are saved in the folder `zuco_preprocessing/results`.

Then they are ready to be finally grouped as .pt, suitable as training/testing/dev data for method1 (folder 'AddToken_Method') and medthod2 (folder 'COGMAP')


# Appendix

## Additional notes:
1. We also computed type-aggregated EEG and gaze data, which are saved in the same folder as the in-context ones 'zuco_sst_rc/results/'. The generating code is in the folder 'zuco_sst_rc/src/zuco_type_normalizing.py', which simply leverages Hollenstein etal. (2019)'s type-aggregation data (.json files) and normalizes the values.

2. NR stands for normal reading from the osf ZUCO folder. 
Two tasks: RC stands for relation classification, and SA stands for sentiment analysis.

3. Clarifiation:
    - IMPORTANT!: in `zuco_preprocessing_SST_RC.py`, We use the preprocessed word-level ET feature values (5 columns representing the 5 gaze feature listed below), instead of raw ET values (4 columns representing pupil_location_X, pupil_location_Y, etc.). Hence, changed dimensions from 4 to 5.
    - hence, in `zuco_utils.py`, the df shape is (1, 5) (1, 104), not (1, 4) (1, 104) in `def get_new_vals(df, mod, t)`


## Data description:
------------------
### EEG:

Each EEG feature, corresponding to the duration of a specific fixation, contains 105 electrode values.

The EEG signal is split into 8 frequency bands, which are fixed ranges of wave frequencies and amplitudes over a time scale$"$:

- $\theta_1$ (4-6Hz)
- $\theta_2$ (6.5-8 Hz)
- $\alpha_1$ (8.5-10 Hz)
- $\alpha_2$ (10.5-13 Hz)
- $\beta_1$ (13.5-18 Hz)
- $\beta_2$ (18.5-30Hz)
- $\gamma_1$ (30.5-40 Hz)
- $\gamma_2$ (40-49.5Hz)

Each EEG feature (corresponding to the duration of a specific fixation) contains 105 electrode values.

The raw EEG data (105 electrode values per word) were averaged (over all fixations for each word) and normalized.


------------------
### Eye-tracking:

- number of fixations (nFix) - the number of all fixations landing on a word;
- first fixation duration (FFD) - the duration of the first fixation on the current word;
- gaze duration (GD) - sum of all fixations on current word in first-pass reading before eye moves out of word;
- go-past time (GPT) - sum of all fixations prior to progressing to the right of current word, including regressions to previous words that originated from the current word.
- total reading time (TRT) - the sum of all fixation durations on the current word;
