from zuco_params import args

ZPTH = "/../content/drive/MyDrive/zuco_sst/data/"
EEG_PTH = f"{ZPTH}type_dict_eeg.json"
ET_PTH = f"{ZPTH}type_dict_gaze.json"
EEG_ET_PTH_NR = f"{ZPTH}zuco-type-aggregation-lexicon-eeg+et8.json"
EEGP_PTH = f"{ZPTH}piece_dict_{args.electrode_handling}_eeg.csv"
ETP_PTH = f"{ZPTH}piece_dict_{args.et_type}_gaze.csv"


# EEG
if args.electrode_handling == 'max':
    AVG_EEG = f"{ZPTH}piece_red_eeg_max_sentences.tsv"
elif args.electrode_handling == 'sum':
    AVG_EEG = f"{ZPTH}piece_red_eeg_sum_sentences.tsv"
else:
    AVG_EEG = f"{ZPTH}piece_red_eeg_sentences.tsv"
# gaze
AVG_ET = f"{ZPTH}piece_red_gaze_sentences.tsv"

# for NR named entity recognition
AVG_EEG_NR = f"{ZPTH}piece_red_eeg_sentences_NR.tsv"
AVG_ET_NR = f"{ZPTH}piece_red_gaze_sentences_NR.tsv"

PHRASE_PTH = f"{ZPTH}phrase_dict.tsv"
