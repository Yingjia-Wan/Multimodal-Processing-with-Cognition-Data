import sklearn
from sklearn.datasets import load_files
import re
import pandas as pd
import numpy as np
import h5py
import sys
import random
import re
import json




def debug(x):
    """Clumsy fn to step through script."""
    if type(x) == tuple:
        for i in x:
            print(f'\n{i}\n')
    else:
        print(f'\n{x}\n')
    sys.exit()

def shuffle_scores(df, t:str = 'agg'):
    """Randomly swap scores for pieces in samples."""
    random.seed(args.seed)
    if t == 'avg':
        random_scores = df.piece_score.values.tolist()
        random.shuffle(random_scores)
        df.piece_score = random_scores
    else:
        random_scores = df[df.score != 0].score.values.tolist()
        random.shuffle(random_scores)
        for idx, ix in enumerate(df[df.score != 0].index):
            df.loc[ix, 'score'] = random_scores[idx]
    return df

def shuffle_scores_b(df, t:str = 'agg'):
    """Randomly sample scores for pieces in samples."""
    random.seed(args.seed)
    if t == 'avg':
        ps_list = df.piece_score.values.tolist()
        mn = min(ps_list)
        mx = max(ps_list)
        random_scores = [random.uniform(mn, mx) for _ in ps_list]
        df.piece_score = random_scores
    else:
        ps_list = df[df.score != 0].score.values.tolist()
        mn = min(ps_list)
        mx = max(ps_list)
        random_scores = [random.uniform(mn, mx) for _ in ps_list]
        for idx, ix in enumerate(df[df.score != 0].index):
            df.loc[ix, 'score'] = random_scores[idx]
    return df
    
# def get_new_vals(df, mod, t):
#     """Convert string to float."""
#     dt = np.float64
#     if t not in ['sum', 'avg']:
#         vals = 'eegvals' if mod == 'eeg' else 'etvals'
#     else:
#         vals = 'avgvals' if t == 'avg' else 'summedvals'
#     new_vals = np.zeros_like(df[vals].values)
#     for ix, val in enumerate(df[vals].values):
#         if "nan" in val:
#             # shape corrected from 4 to 5 - YW
#             if mod == "gaze":
#                 shape = (1, 5)
#             else:
#                 shape = (1, 104)
#             v = np.zeros(shape=shape)
#         else:
#             val = val.replace("\n", " ")
#             valr = re.sub(r" +", r", ", val) # replace multiple spaces with comma
#             v = eval(valr.replace("[,", "[")) # text to list
#         new_vals[ix] = np.array(v, dtype=dt)
#     return new_vals

def get_new_vals(df, mod, t):
    #Convert string to float.
    dt = np.float64
    if t in ['sum', 'avg']:
        vals = 'avgvals' if t == 'avg' else 'summedvals'
    else:
        if mod == 'eeg':
            vals = 'eegvals'
            new_vals = np.zeros_like(df[vals].values)
            for ix, val in enumerate(df[vals].values):
                if "nan" in val:
                    shape = (1, 104)
                    v = np.zeros(shape=shape)
                else: # turns the values of avgvals to actual list of 104 values
                    val = val.replace("\n", " ")
                    valr = re.sub(r" +", r", ", val) # replace multiple spaces with comma
                    v = eval(valr.replace("[,", "[")) # text to list
                new_vals[ix] = np.array(v, dtype=dt)
        else:
            print("mod is not eeg. Please check.")
    return new_vals

def get_new_vals_hot(df, mod, t):
    """Convert string to float."""
    dt = np.int64
    vals = 'label_one_hot'
    new_vals = np.zeros_like(df[vals].values)
    for ix, val in enumerate(df[vals].values):
        if type(val) == float or "nan" in val:
            shape = (1, 11)
            v = np.zeros(shape=shape)
        else:
            val = val.replace("\n", " ")
            valr = re.sub(r" +", r", ", val)
            v = eval(valr.replace("[,", "["))
        new_vals[ix] = np.array(v, dtype=dt)
    return new_vals

def load_df(pth, mod, t: str = ""):
    df = pd.read_csv(pth, sep="\t")
    try:
        # df = df.drop(columns='Unnamed: 0')
        df = df.iloc[:, 1:] # drop the first column; this doesn't change the original csv file.
    except:
        print('Error in dropping the first column in load_df.')
    if mod == "eeg":
        if not "piece" in pth:
            # check your own eeg.tsv file to see if the column names are correct - YW
            df.iloc[:, 3] = get_new_vals(df, mod, t) # modify the 4th column: original word values to input ids
        else:
            df.iloc[:, 6] = get_new_vals(df, mod, t) # modify 7th: avgvals to np.array
            df.iloc[:, 9] = get_new_vals_hot(df, mod, "pieced") # 10th: label_one_hot
    return df

def read_config_file(filename):
    """Read current config file"""
    with open(filename, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    return cfg


################################################### preprocessing text data  #################################################
#@title Secondary
def open_subject_sentence_data(subject):
    filepath = f"/../../sentiment-analysis/Data_to_preprocess/results{subject}_NR.mat"
    f = h5py.File(filepath, "r+")
    return f

def get_sentence_order(dataset, task):
    # Find numbers between '/' and '.txt' in filenames, convert them to integers and return a list
    # returning the list of the numbers before _. If there are repetitive integers in the filenames, they will appear multiple times in the resulting list.
    if task == "SA":
        return [int(filename.split("/")[-1].split(".")[0]) for filename in dataset["filenames"]]
    if task == "RC":
        return [int(filename.split("/")[-1].split("_")[0]) for filename in dataset["filenames"]]

def load_data_labels(dataset):
    # Split by words
    x_text = dataset['data']
    x_text = [clean_str(sent) for sent in x_text] # general word split rules
    # Generate labels
    labels = []
    for i in range(len(x_text)):
        label = [0 for j in dataset['target_names']]
        label[dataset['target'][i]] = 1
        labels.append(label)
    y = np.array(labels)
    return [x_text, y]

def clean_str(string):
    string = string.replace(".", "")
    string = string.replace(",", "")
    string = string.replace("--", "")
    string = string.replace("`", "")
    string = string.replace("''", "")
    string = string.replace("' ", " ")
    string = string.replace("*", "")
    string = string.replace("\\", "")
    string = string.replace(";", "")
    string = string.replace("- ", " ")
    string = string.replace("/", "-")
    string = string.replace("!", "")
    string = string.replace("?", "")
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()



def clean_str_RC(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def get_processed_dataset_SA(dataset_path, binary, verbose, labels_from):
    """
    Load and process raw dataframes to a usable format for training
    :param dataset_path:    (str)   Path to the dataset to process
    :param binary:          (bool)  Output sentiment-binary format
    :param verbose:         (bool)  Verbose output
    :param labels_from:     (str)   Name of the subject from whom to take the labels
    :return:
        dataset:            (~dic)  Dictionary-like object containing the data from dataset_path
        x:                  (array) Array of integers, ncol = max phrase length, nrow = number of sentences, each value being the vocab index of the c-th word in the r-th sentence
        x_text:             (list)  List of all sentences
        y:                  (list)  List of lists, each sublist is the OHE vector of the response variable in the 47 control items
        vocab_processor:    (tfobj) Converts word idx to word and vice versa
    """

    categories = ["NEGATIVE", "POSITIVE"] if binary else None
    subfolder_name = labels_from or "all"

    dataset = load_files(container_path=dataset_path + "/" + subfolder_name, categories=categories,
                          load_content=True, shuffle=False, encoding='utf-8')

    x_text, y = load_data_labels(dataset)

    # Build vocabulary
    max_sentence_length = max([len(x.split(" ")) for x in x_text])

    return dataset, x_text, y

def get_processed_dataset_RC(dataset_path, binary, verbose, labels_from):
    '''
    :return:
        datasets: dictionary-like object; containing following attributes:
            data: list of str; The raw text data to learn.
            target: nparray; The target labels (integer index).
            target_names: list; The names of target classes.
            filenames: nparray;
            DESCRstr: str; The full description of the dataset.
        x_text: (list)  List of all sentences
        y: label index
        x_text_entities: (list) List of all entities with tags
    '''
    categories = [  "AWARDED",
                    "EDUCATION",
                    "EMPLOYER",
                    "FOUNDER",
                    "JOBTITLE",
                    "NATIONALITY",
                    "POLITICALAFFILIATION",
                    "VISITED",
                    "WIFE",
                    "BIRTHPLACE",
                    "DEATHPLACE"]
    datasets = get_datasets_zuco(container_path=dataset_path, categories=categories)
    x_text, y = load_data_labels(datasets)

    # transform x_text to x_text_entities
        # iterate over each sentence, split words into list t, find index of <e> and <-e> as tags
    x_text_split = [t.split(' ') for t in x_text]
    opening_tags = [[i for i, j in enumerate(t) if j == '<e>'] for t in x_text_split]
    closing_tags = [[i for i, j in enumerate(t) if j == '<-e>'] for t in x_text_split]
    x_text_entities = []
        # join words between tags, append to x_text_entities which is a list of entities with tags
    for t, o, c in zip(x_text_split, opening_tags, closing_tags):
        # Without tags
        # x_text_entities.append(t[(o[0]+1):c[0]]+t[(o[1]+1):c[1]])
        # With tags
        x_text_entities.append( " ".join( t[o[0]:(c[0]+1)] + t[o[1]:(c[1]+1)] ) )

    return datasets, x_text, y, x_text_entities

def get_datasets_zuco(container_path=None, categories=None, load_content=True,
                       encoding='utf-8'):
    """
    Load text files with categories as subfolder names.
    Individual samples are assumed to be files stored a two levels folder structure.
    :param container_path: The path of the container
    :param categories: List of classes to choose, all classes are chosen by default (if empty or omitted)
    :param shuffle: shuffle the list or not
    :param random_state: seed integer to shuffle the dataset
    :return: dictionary-like object; data and labels of the dataset
    """

    # load_files from sklearn.datasets: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_files.html
    datasets = load_files(container_path=container_path, categories=categories,
                          load_content=load_content, shuffle=False, encoding=encoding)
    return datasets

def reshape_sentence_features(sentence_features, max_seq_length_text):
    # Pad features with zeros given the different sentence lengths
    max_length = np.max([sf.shape[0] for sf in sentence_features])
    if max_seq_length_text > max_length:
        max_length = max_seq_length_text
    padded_sentences = [np.pad(sf, ((0, max_length-sf.shape[0]),(0,0)), 'constant') for sf in sentence_features]
    return np.stack(padded_sentences, axis=0)


def get_word_features(word):
    if type(word.FFD) == int:
        return np.array([word.nFixations, word.FFD, word.GD,
                         word.GPT, word.TRT])
    return np.zeros(N_ET_FEATURES)


def load_entities_by_file(file_path):
    with open(file_path, encoding="utf-8", mode='r') as entities_by_file_number_json_file:
        return json.load(entities_by_file_number_json_file)

def get_entity_ids(cfg, dataset_name, filenames):
    entities_by_file = load_entities_by_file(cfg["datasets"][dataset_name]["entities_file_path"])
    return [entities_by_file[x] for x in
                  [re.findall('/([0-9]+_[0-9]+).txt$', filename)[0]
                   for filename in filenames]]






# # TODO: see zuco-nlp/sentiment-analysis/data_creation_utils.py - YW

'''
def load_matlab_string(matlab_extracted_object):
    extracted_string = u''.join(chr(c) for c in matlab_extracted_object)
    return extracted_string

def extract_all_fixations(data_container, word_data_object, float_resolution = np.float16):
    word_data = data_container[word_data_object]
    fixations_data = {}
    if len(word_data.shape) > 1:
        for fixation_idx in range(word_data.shape[0]):
            fixations_data[fixation_idx] = np.array(data_container[word_data[fixation_idx][0]]).astype(float_resolution)
    return fixations_data

def extract_word_order_from_fixations(fixations_order_per_word):
    if not fixations_order_per_word:
        return []
    fxs_list = [list(fixs) if len(fixs.shape)>1 else [] for fixs in fixations_order_per_word]
    n_tot_fixations = len(sum(fxs_list, []))
    words_fixated_in_order = []
    for fixation_n in range(n_tot_fixations):
        mins_per_word_idx = np.array([min(i) if len(i)>0 else np.nan for i in fxs_list])
        next_word_fixated = int(np.nanargmin(mins_per_word_idx)) # Seems to work like this
        fxs_list[next_word_fixated].remove(min(fxs_list[next_word_fixated]))
        words_fixated_in_order.append(next_word_fixated)
    return words_fixated_in_order



def extract_sentence_level_data(subject, eeg_float_resolution=np.float16):
    f = open_subject_sentence_data(subject)
    sentence_data = f['sentenceData']
    rawData = sentence_data['rawData']
    icaData = sentence_data['IC_act_automagic']
    contentData = sentence_data['content']
    wordData = sentence_data['word']

    #dataset_path = "/../content/drive/My Drive/brain-decoding/zuco-nlp/sentiment-analysis/data/sentences"
    dataset_path = "/../content/zuco-nlp/relation-classification/data/zuco_nr_cleanphrases/"
    dataset, x_text, y, x_text_entities = get_processed_dataset_NR(dataset_path=dataset_path,
                                                                   binary=False, verbose=True,
                                                                   labels_from=None)
    #sentence_order = get_sentence_order(dataset)
    sentence_level_data = {}
    for idx in range(len(rawData)): # raw data is an used but they all should be the same in length (400 for ternary, about 2/3 of that for binary)
        data_dict = {}
        obj_reference_raw = rawData[idx][0]
        data_dict["RAW_EEG"] = np.array(f[obj_reference_raw]).astype(eeg_float_resolution)
        obj_reference_ica = icaData[idx][0]
        data_dict["ICA_EEG"] = np.array(f[obj_reference_ica]).astype(eeg_float_resolution)
        obj_reference_content = contentData[idx][0]
        data_dict["content"] = load_matlab_string(f[obj_reference_content])
        # do_print(data_dict["content"], report_file)
        data_dict["sentence_number"] = idx
        #label_idx = np.where(np.array(sentence_order) == idx)[0][0]
        #data_dict["label"] = np.array(y[label_idx])
        #data_dict["entities"] = np.array(x_text_entities[label_idx])
        #data_dict["word_embedding_idxs"] = np.array(x[label_idx, :])
        #data_dict["label_content"] = dataset['data'][label_idx]
        #label_n = np.where(data_dict["label"] == 1)[0][0]
        #data_dict["label_name"] = dataset['target_names'][label_n]

        #bad_channels = get_bad_channels(idx, subject)
        #data_dict["bad_channels"] = bad_channels.split(" ") if type(bad_channels) == str else None
        try:
            data_dict["word_level_data"] = extract_word_level_data(idx, f, f[wordData[idx][0]], eeg_float_resolution=eeg_float_resolution)
        except ValueError:
            data_dict["word_level_data"] = dict()
        sentence_level_data[idx] = data_dict
    f.close()
    return sentence_level_data

def create_all_subjects_data(filename, subject, eeg_float_resolution=np.float16):
    if not os.path.exists(filename):
        os.mkdir(filename)
    all_subjects_dict = {}
    if subject:
        subjectnames = [subject]
    else:
        subjectnames = SUBJECT_NAMES # TODO: ZDN (currently error exists)
    for subject in subjectnames:
        all_sentences_info = extract_sentence_level_data(subject, eeg_float_resolution=eeg_float_resolution)
        all_subjects_dict[subject] = all_sentences_info
        subject_file = filename + "_" + subject + ".pickle"
        print("Data saved in file " + subject_file)
        with open(subject_file, "wb") as f:
            pkl.dump(all_sentences_info, f)
    return all_subjects_dict


def extract_word_level_data(idx, data_container, word_objects, eeg_float_resolution = np.float16):
    available_objects = list(word_objects)
    contentData = word_objects['content']
    fixations_order_per_word = []
    if "rawEEG" in available_objects:
        rawData = word_objects['rawEEG']
        icaData = word_objects['IC_act_automagic']
        etData = word_objects['rawET']
        # TODO: Double check that this works
        ffdData = word_objects['FFD']
        gdData = word_objects['GD']
        gptData = word_objects['GPT']
        trtData = word_objects['TRT']
        nFixData = word_objects['nFixations']
        fixPositions = word_objects["fixPositions"]
        assert len(contentData) == len(etData) == len(icaData) == len(rawData), "different amounts of different data!!"

        zipped_data = zip(rawData, icaData, etData, contentData, ffdData, gdData, gptData, trtData, nFixData, fixPositions)
        word_level_data = {}
        word_idx = 0
        for raw_eegs_obj, ica_eegs_obj, ets_obj, word_obj, ffd, gd, gpt, trt, nFix, fixPos in zipped_data:
            word_string = load_matlab_string(data_container[word_obj[0]])
            if is_real_word(word_string):
                data_dict = {}
                data_dict["RAW_EEG"] = extract_all_fixations(data_container, raw_eegs_obj[0], eeg_float_resolution)
                data_dict["ICA_EEG"] = extract_all_fixations(data_container, ica_eegs_obj[0], eeg_float_resolution)
                data_dict["RAW_ET"] = extract_all_fixations(data_container, ets_obj[0], np.float32)
                # TODO: Fill the following in!!! <---- IMPORTANT TO DO ASAP
                data_dict["FFD"] = data_container[ffd[0]].value[0, 0] if len(data_container[ffd[0]].value.shape) == 2 else None
                data_dict["GD"] = data_container[gd[0]].value[0, 0] if len(data_container[gd[0]].value.shape) == 2 else None
                data_dict["GPT"] = data_container[gpt[0]].value[0, 0] if len(data_container[gpt[0]].value.shape) == 2 else None
                data_dict["TRT"] = data_container[trt[0]].value[0, 0] if len(data_container[trt[0]].value.shape) == 2 else None
                data_dict["nFix"] = data_container[nFix[0]].value[0, 0] if len(data_container[nFix[0]].value.shape) == 2 else None
                fixations_order_per_word.append(fixPos)

                data_dict["word_idx"] = word_idx
                # TODO: data_dict["word2vec_idx"] = Looked up after through the actual word.
                data_dict["content"] = word_string
                word_idx += 1
                word_level_data[word_idx] = data_dict
            else:
                print(word_string + " is not a real word.")
    else:
        # If there are no word-level data it will be word embeddings alone
        word_level_data = {}
        word_idx = 0
        for word_obj in contentData:
            word_string = load_matlab_string(data_container[word_obj[0]])
            if is_real_word(word_string):
                data_dict = {}
                #TODO: Make sure it was a good call to convert the below from {} to None
                data_dict["RAW_EEG"] = {}
                data_dict["ICA_EEG"] = {}
                data_dict["RAW_ET"] = {}
                data_dict["FFD"] = None
                data_dict["GD"] = None
                data_dict["GPT"] = None
                data_dict["TRT"] = None
                data_dict["nFix"] = None
                data_dict["word_idx"] = word_idx
                data_dict["content"] = word_string
                word_level_data[word_idx] = data_dict
                word_idx += 1
            else:
                print(word_string + " is not a real word.")
        sentence = " ".join([load_matlab_string(data_container[word_obj[0]]) for word_obj in word_objects['content']])
        print("Only available objects for the sentence '{}' are {}.".format(sentence, available_objects))
    word_level_data["word_reading_order"] = extract_word_order_from_fixations(fixations_order_per_word)
    return word_level_data




def get_et_features_single_subject(subject, file_name, max_sequence_length):
    f = open_subject_sentence_data(subject)
    data = f['sentenceData']
    rawData = data['rawData']
    contentData = data['content']
    wordData = data['word']
    for idx in range(len(rawData)):
        data_dict = {}
        obj_reference_raw = rawData[idx][0]
        obj_reference_content = contentData[idx][0]
        data_dict["content"] = load_matlab_string(f[obj_reference_content])
        data_dict["sentence_number"] = idx
    sentence_features = [np.array([get_word_features(word)
                                   for word in sentence.word
                                   if is_real_word(word)])
                         for sentence in data]
    sentence_features = np.array(sentence_features)

    return reshape_sentence_features(sentence_features, max_sequence_length)

def get_eye_tracking_features(file_name, subject, max_sequence_length):
    if not os.path.exists(file_name):
        os.mkdir(file_name)
    et_features_by_subject = {}
    if subject:
        subject_names = [subject]
    else:
        subject_names = SUBJECT_NAMES
    for subject in subject_names:
        print("Reading ET data for subject: " + subject)
        et_features_by_subject[subject] = get_et_features_single_subject(subject, file_name, max_sequence_length)
        subject_file = file_name + "_" + subject + "_ET.pickle"
        print("ET Data saved in file " + subject_file)
        with open(subject_file, "wb") as f:
            pkl.dump(et_features_by_subject, f)
    return et_features_by_subject




'''

