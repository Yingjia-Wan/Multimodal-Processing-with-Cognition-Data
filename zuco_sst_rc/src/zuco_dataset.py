import torch
from zuco_paths import *
import pandas as pd
import numpy as np
from zuco_utils import *
from zuco_params import args

normEEG = ((args.electrode_handling in ['max', 'sum']  and
            args.per_sample) or
            (not args.per_sample and
            args.electrode_handling == 'sum'))

class ZuCoDataset(torch.utils.data.Dataset):
    """Load ZuCo sentiment dataset."""
    def __init__(self, encodings, labels, split = '', new2old = dict()):
        self.encodings = encodings
        self.new2old = new2old
        self.labels = labels
        self.split = split
        self.zuco_eeg = pd.read_csv(EEGP_PTH)
        self.zuco_et = pd.read_csv(ETP_PTH)
        self.phrase_dict = pd.read_csv(PHRASE_PTH, sep="\t")
        if args.task == 'sst':
            self.avg_eeg = pd.read_csv(AVG_EEG, sep="\t")
            self.avg_et = pd.read_csv(AVG_ET, sep="\t")
        elif args.task == 'rel':
            self.avg_eeg = pd.read_csv(AVG_EEG_NR, sep="\t")
            self.avg_et = pd.read_csv(AVG_ET_NR, sep="\t")
        if args.random_scores:
            self.zuco_eeg = shuffle_scores_b(self.zuco_eeg)
            self.zuco_et = shuffle_scores_b(self.zuco_et)
            self.avg_eeg = shuffle_scores_b(self.avg_eeg, 'avg')
            self.avg_et = shuffle_scores_b(self.avg_et, 'avg')

        # Producing avgvals and label_one_hot for each sentence
        # 'avg'
        self.avg_eeg.avgvals = get_new_vals(self.avg_eeg, "eeg", "avg")
            # 'et' changed to 'gaze' for consistency - YW
        self.avg_et.avgvals = get_new_vals(self.avg_et, "gaze", "avg")
        # 'pieced'
        self.avg_eeg.label_one_hot = get_new_vals_hot(self.avg_eeg,
                                                      'eeg', "pieced")
        self.avg_et.label_one_hot = get_new_vals_hot(self.avg_et,
                                                     'gaze', "pieced")

    def __getitem__(self, idx):
        item = {key: val[idx]
                for key, val in self.encodings.items()}
        if 'zuco' not in args.run_name and (args.save_att or self.split == 'train'):
            ids = self.encodings['input_ids'][idx]
            # Obtain dataframe entries for sentence # idx
            if args.task == 'sst':
                avg_eeg = self.avg_eeg[self.avg_eeg.sent == self.new2old[idx]]
                avg_et = self.avg_et[self.avg_et.sent == self.new2old[idx]]
            elif args.task == 'rel':
                avg_eeg = self.avg_eeg[self.avg_eeg.sent == self.new2old[idx]]
                avg_et = self.avg_et[self.avg_et.sent == self.new2old[idx]]

            eeg_scores = []
            et_scores = []

            for id in ids:
                try:
                    eeg_scores.append(self.zuco_eeg[self.zuco_eeg.id == id.item()].score.values.tolist()[0])
                except IndexError:
                    eeg_scores.append(0)
                try:
                    et_scores.append(self.zuco_et[self.zuco_et.id == id.item()].score.values.tolist()[0])
                except IndexError:
                    et_scores.append(0)

            # 'red' for reduced
            red_eeg_scores = []
            red_et_scores = []
            for id in ids:
                temp_eeg = avg_eeg[avg_eeg.input_id == id.item()]
                temp_et = avg_et[avg_et.input_id == id.item()]
                if id.item() not in [30522, 30523]:
                    try:
                        red_eeg_scores.append(temp_eeg.piece_score.values.tolist()[0])
                    except IndexError:
                        if id.item() not in [101, 102, 0]:
                            red_eeg_scores.append(0)
                    try:
                        red_et_scores.append(temp_et.piece_score.values.tolist()[0])
                    except IndexError:
                        if id.item() not in [101, 102, 0]:
                            red_et_scores.append(0)
                elif id.item() in [30522, 30523]:
                    red_eeg_scores.append(0)
                    red_et_scores.append(0)

            #red_eeg_embeds = avg_eeg.avgvals.values.tolist()
            #red_et_embeds = avg_et.avgvals.values.tolist()

            eeg_ids = avg_eeg.input_id.values.tolist()
            et_ids = avg_et.input_id.values.tolist()

            # Double quotes
            if eeg_ids[0] == 1000 and et_ids[-1] == 1000:
                red_eeg_scores = red_eeg_scores[1:-1]
                #red_eeg_embeds = red_eeg_embeds[1:-1]
                red_et_scores = red_et_scores[1:-1]
                #red_et_embeds = red_et_embeds[1:-1]

            maxnorm = lambda t: t/(max(1e-12, t.max()))

            red_et_scores = maxnorm(np.array(red_et_scores)).tolist()
            if normEEG:
                red_eeg_scores = maxnorm(np.array(red_eeg_scores)).tolist()

            # [CLS], [SEP]
            red_eeg_scores = [0.] + red_eeg_scores + [0.]
            red_et_scores = [0.] + red_et_scores + [0.]
            """
            red_eeg_embeds = [np.zeros_like(red_eeg_embeds[0])] + \
                              red_eeg_embeds + \
                             [np.zeros_like(red_eeg_embeds[0])]
            red_et_embeds = [np.zeros_like(red_et_embeds[0])] + \
                             red_et_embeds + \
                             [np.zeros_like(red_et_embeds[0])]
            """

            # [PAD]

            red_eeg_scores += [0. for _ in range(ids.tolist().count(0))]
            red_et_scores += [0. for _ in range(ids.tolist().count(0))]
            """
            red_eeg_embeds += [np.zeros_like(red_eeg_embeds[0])
                               for _ in range(ids.tolist().count(0))]
            red_et_embeds += [np.zeros_like(red_et_embeds[0])
                              for _ in range(ids.tolist().count(0))]
            """

            if args.task == "sst":
                item['eeg'] = torch.tensor(eeg_scores)
                item['et'] = torch.tensor(et_scores)
            item['eeg_redmn'] = torch.tensor(red_eeg_scores)
            item['et_trt'] = torch.tensor(red_et_scores)
            #item['eeg_embeds'] = torch.tensor(red_eeg_embeds).float()
            #item['et_embeds'] = torch.tensor(red_et_embeds).float()

        item['labels'] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.labels)

class SSTDataset(torch.utils.data.Dataset):
    """Load SST-3 dataset minus ZuCo."""
    def __init__(self, encodings, labels, config, split = ''):
        self.encodings = encodings
        self.labels = labels
        self.split = split
        self.zuco_eeg = pd.read_csv(EEGP_PTH)
        self.zuco_et = pd.read_csv(ETP_PTH)
        if args.random_scores:
            self.zuco_eeg = shuffle_scores_b(self.zuco_eeg)
            self.zuco_et = shuffle_scores_b(self.zuco_et)
        self.class2idx = config.label2id

    def __getitem__(self, idx):
        item = {key: val[idx]
                for key, val in self.encodings.items()}
        if 'zuco' not in args.run_name and (args.save_att or self.split == 'train'):
            ids = self.encodings['input_ids'][idx]
            eeg_scores = []
            et_scores = []
            for child_ix, id in enumerate(ids):
                try:
                    eeg_scores.append(self.zuco_eeg[self.zuco_eeg.id == id.item()].score.values.tolist()[0])
                except IndexError:
                    eeg_scores.append(0)
                try:
                    et_scores.append(self.zuco_et[self.zuco_et.id == id.item()].score.values.tolist()[0])
                except IndexError:
                    et_scores.append(0)
            item['eeg'] = torch.tensor(eeg_scores)
            item['et'] = torch.tensor(et_scores)
        item['labels'] = torch.tensor(self.class2idx[self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)
