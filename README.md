# Multimodal-Processing-with-Cognition-Data
Implementation code for my Master's Dissertation: "Multimodal Prompt-Tuning with Human Cognition Data" by Yingjia Wan.

# Data Source
- The original ZuCo corpus is available at https://osf.io/uxamg/. As the data files were large, I recommend downloading them efficiently using my code at `./zuco_preprocessing/src/zuco_matfiles_download.py`. See systematic instructions below.

- This project also requires a `zuco_data_storage` folder inside the current dir, which is imported from https://github.com/DS3Lab/zuco-nlp/tree/master by Nora Hollenstein. I rename the folder name `zuco-nlp` to `zuco_data_storage`. See [README.md](./zuco_data_storage/README.md) in the `zuco_data_storage` folder for details.


# Instructions

Please refer to the subsequent README.md files in each folder for the instructions of setting up the environment for each step.

## 1. Data Preprocessing
First, please refer to the [README.md](./zuco_preprocessing/README.md) in the folder `zuco_preprocessing` for data formatting and preprocessing instructions.
## 2. Experiment 1
Please refer to the [README.md]() in the folder `AddToken_Method` for the instructions of Experiment 1.
## 3. Experiment 2
Please refer to the [README.md](./CogMAP/README.md) in the folder `COGMAP` for the instructions of Experiment 2.


