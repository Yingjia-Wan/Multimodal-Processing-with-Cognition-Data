# Overview
The `zuco_preprocessing` section aims to extract task-specific word-level gaze and EEG data from the ZUCO corpus. It preprocesses the .mat data to .tsv files by performing downloading, organizing on the word-level, converting file formats, etc. The original ZuCo corpus is available at https://osf.io/uxamg/.

# Instructions for data preprocessing:
### 1. Download:
Download matlab files from ZuCo (over 50 GB) efficiently by running `zuco_matfiles_download.py` in the folder `zuco_preprocessing/src`. The matfiles will be saved in the folder `zuco_data_storage/{SUBJECT_ID}/Data_to_preprocess`. (Mannual download is not recommended due to the large size of the files.)

### 2. Convert:
Convert the matfiles into pkl files, by running `create_modeling_data.py` in the folder e.g, `zuco_data_storage/sentiment-analysis/`.
This will create the pkl files for each SUBJECT_ID in the subfolder `../Result_files`

### 3. Merge:

Merge the subject-pkl files into csv files averaging over all subjects, create word-level gaze and EEG datasets **across subjects**, by running `zuco_preprocessing/src/zuco_preprocessing_SST_RC.py`. The result avg.tsv files are saved in the folder `zuco_preprocessing/results`.

Then they are ready to be finally grouped as .pt, suitable as training/testing/dev data for method1 ('AddToken_Method') and medthod2 ('COGMAP').

### Additional notes:
1. NR stands for normal reading from the osf ZUCO folder; RC for relation classification; SA for sentiment analysis.

3. In `zuco_preprocessing_SST_RC.py`, We use the preprocessed word-level ET feature values (5 columns representing the 5 gaze feature listed below), instead of raw ET values (4 columns representing pupil_location_X, pupil_location_Y, etc.).

# Data description:
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
