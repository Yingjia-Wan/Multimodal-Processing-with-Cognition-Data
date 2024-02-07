# Acknowledgement
The zuco_sst_rc data folder preprocesses the .mat data to .tsv files by performing downloading, organizing on the word-level, converting file formats, etc.

- This project folder 'zuco_sst' derives from https://drive.google.com/drive/folders/1_zJRdPai1Y9NQkutPvqjUwb2Rs0qt6l1?usp=drive_link, and can be used to derive task-specific word-level gaze and EEG datakets from the ZUCO dataset. However, another execution python file is added in the dir as the executing file (), which is also referenced from Erick Mcguire: https://colab.research.google.com/drive/1q6InysgFbFq5I2ULuBa91krkktvcJ4fe?usp=sharing, but with extensive changes made as addressed below.

- The original ZuCo dataset is available at https://osf.io/uxamg/

- This project also requires a zuco-nlp folder parallel to the current dir (zuco_sst_rc), which is downloaded from https://github.com/DS3Lab/zuco-nlp/tree/master

# Instructions for data preprocessing:
1. download matlab files (very large) from ZuCo by running code in the folder `zuco_sst_rc/src`: `zuco_matfiles_download.py`. The matfiles are saved in the folder `zuco_nlp/XXXXX/Data_to_preprocess`.


2. transform the matfiles into pkl files, using code in the folder e.g, zuco_nlp/sentiment-analysis/: `create_modeling_data.py`.
This will create the pkl files for each subject in the subfolder `/Result_files`

3. transform the subject-pkl files into csv files, averaging over all subjects and create word-level gaze and EEG datasets, using code in the folder `zuco_sst_rc/src`:
run: `zuco_preprocessing_SST_RC.py`.
The result avg.tsv files are saved in the folder `zuco_sst_rc/results`.

Then they are ready to be finally grouped as .pt, suitable as training/testing/dev data for method1 (folder 'AddToken_Method') and medthod2 (folder 'COGMAP')


# Appendix

## Additional notes:
1. We also computed type-aggregated EEG and gaze data, which are saved in the same folder as the in-context ones 'zuco_sst_rc/results/'. The generating code is in the folder 'zuco_sst_rc/src/zuco_type_normalizing.py', which simply leverages Hollenstein etal. (2019)'s type-aggregation data (.json files) and normalizes the values.

2. NR stands for normal reading from the osf ZUCO folder. 
Two tasks: RC stands for relation classification, and SA stands for sentiment analysis.

## Declaration: zuco_sst_rc major changes:
1. changes in zuco_dataset.py
    - 'et' renamed to 'gaze' for var and argument consistency: self.avg_et.avgvals = get_new_vals(self.avg_et, "gaze", "avg")
2. changes in zuco_preprocessing_SST_RC.py
    - IMPORTANT!: We use the already preprocessed word-level ET feature values (5 columns), instead of raw ET values (4 columns). Hence, changed dimensions from 4 to 5.
3. Changes in zuco_utils.py
    - hence, changed the df shape from (1, 4) (1, 104) to (1, 5) (1, 104): def get_new_vals(df, mod, t)


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
