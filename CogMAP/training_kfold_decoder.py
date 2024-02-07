'''
The training code for CogMAPL is written by Yingjia Wan, University of Cambridge, 2023

Useful training tutorial: 
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning#training
https://gmihaila.github.io/tutorial_notebooks/gpt2_finetune_classification/
GPT2ForSequenceClassification:
https://huggingface.co/transformers/v4.8.0/_modules/transformers/models/gpt2/modeling_gpt2.html#GPT2ForSequenceClassification
https://huggingface.co/transformers/v4.8.0/model_doc/gpt2.html#gpt2forsequenceclassification

try run training using Trainer. 
https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer
https://huggingface.co/docs/transformers/main/tasks/sequence_classification#evaluate

Notes:
1. training data of numpy arrays should be stored to .pt (pytorch tensor file) instead of .json file which is typically for textual input. (because json file by default convert the values to strings)
2. align the ground_truth format with it
3. not only model, but data is also needed to be moved to device
4. decoder model: compare what is inputed in the loss function: embedding? texts?

'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from COGMAPL_decoder import COGMAPL, LanguageDecoder, MappingNetwork
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          AutoConfig,
                          AutoTokenizer,
                          AdamW, 
                          get_linear_schedule_with_warmup,
                          AutoModelForSequenceClassification)
from tqdm.notebook import tqdm
from ml_things import plot_dict, plot_confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
import datasets
import os
import argparse
import json
import pickle
from utils import *
from sklearn.model_selection import StratifiedKFold
import constants

def parse_args():
  # Create parser
  parser = argparse.ArgumentParser(description='Hyperparameters for model training')

  # Add arguments
  parser.add_argument('--model_name_or_path', type=str, required=True)
  parser.add_argument('--cognition_type', type=str, required=True)
  parser.add_argument('--batch_size', type=int, required=True)
  parser.add_argument('--learning_rate', type=float, required=True)
  parser.add_argument('--num_epochs', type=int, required=True)
  parser.add_argument('--max_sequence_len', type=int, required=False)
  parser.add_argument('--seed', type=int, required=True)

  # Parse arguments
  args = parser.parse_args()
  return args

########################################################### Hyperparameter Set up ###########################################################
# parse arguments to get hyperparameters for training.sh
args = parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set seed for reproducibility.
seed = args.seed
set_seed(seed)

# Set model_name, and cognition_type
model_name_or_path = args.model_name_or_path #'gpt2' or 'gpt2-medium' or 'gpt2-large' or 'gpt2-xl' or 'EleutherAI/gpt-neo-125M' or 'EleutherAI/gpt-neo-1.3B' or 'facebook/opt-350m'
cognition_type = args.cognition_type # 'ET' or 'EEG' or 'ET+EEG' or 'ET_ZAB' or 'ET_ZAB+EEG'
task = 'SA' # 'SA' or 'RC'

# training hyperparameters
batch_size = args.batch_size # 32
learning_rate = args.learning_rate # best so far is 5e-5 for ET
learning_rate_str = "{:.0e}".format(learning_rate)  # Format the learning rate as '1e-4' for saving the directory name
num_epochs = args.num_epochs # 10
max_sequence_len = args.max_sequence_len if args.max_sequence_len is not None else None # GPT2 default is 1024
# optional: the output_length of the mapper. It is set conditioned on cognition_type in COGMAPL.py.

# label2id
labels_ids = {'negative': 0, 'neutral':1, 'positive': 2}
n_labels = len(labels_ids) # This is used to decide size of classification head.


# Get input data for training
train_file = f'./data/{cognition_type}/SA/kfold_train_raw_data.pt'
test_file = f'./data/{cognition_type}/SA/kfold_test_raw_data.pt'

############################################ Load the model and tokenizer  ################################################################
# Get the actual model: COGMAPL
print(f'Loading model {model_name_or_path} for task {task} on {cognition_type}, {num_epochs} epochs, lr {learning_rate_str}...')
model = COGMAPL(model_id = model_name_or_path, 
                cognition_type = cognition_type, 
                num_labels = n_labels).to(device)

# Get model's tokenizer.
tokenizer = model.text_processor # Pading is set from left within the model.

# Define PAD Token = EOS Token = 50256
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

########################################### Define MyDataset and data_collator ##############################################################

# Set up self-defined collator for padding and batch processing.
gpt2_classificaiton_collator = datasets.Gpt2ClassificationCollator(use_tokenizer=tokenizer, 
                                                          labels_encoder=labels_ids, pad_cognition=True, max_seq_len = max_sequence_len)

# Custom dataset class for loading the data
class MyDataset(Dataset):
    def __init__(self, data_file):
        # Load the data from the file
        self.data = torch.load(data_file)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        # get the data, later to be fed into the data loader
        cognition = sample['cognition']
        sentence = sample['sentence']
        label = sample['label']
        input = [cognition, sentence, label] # order matters for data_collator
        
        return {'input': input, 'sentence': sentence, 'cognition': cognition, 'label': label}


############################################### Hyperparameter Search ##############################################################
# https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.hyperparameter_search









########################################################### Run Training  ##############################################################

# Set up k-fold cross-validation
k_folds = constants.K_FOLDS # 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=constants.SPLIT_SEED) # spliting seed = 42, different from the seed for training

# Define training arguments: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments

training_args = TrainingArguments(
    output_dir= f'./results/{model_name_or_path}/{cognition_type}/lr_{learning_rate_str}_bs_{batch_size}_epochs_{num_epochs}/seed_{seed}', # output directory
    num_train_epochs= num_epochs, # 20
    per_device_train_batch_size= batch_size, # 32
    learning_rate= learning_rate, # 5e-5

    save_steps=500, # Number of updates steps before checkpoint saves if save_strategy="steps".
    evaluation_strategy="steps",
    eval_steps=500,
    save_total_limit=1,
    load_best_model_at_end=True # Save only the best model
)


# innitialize aggregation lists for all folds
all_test_labels = []
all_test_predictions = []
all_fold_test_accuracies = []

# Prepare the training data for k-fold cross-validation
train_dataset = MyDataset(train_file)
X_train = [sample['input'] for sample in train_dataset]
y_train = [sample['label'] for sample in train_dataset]
# prepare the test data separately
test_dataset = MyDataset(test_file)
X_test = [sample['input'] for sample in test_dataset]
y_test = [sample['label'] for sample in test_dataset]


y_test_ids = []
for label_str in y_test:
    label_id = labels_ids[label_str]
    y_test_ids.append(label_id)

# Loop over k folds
for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
    # print(type(train_index))
    # print(train_index)
    # print(type(fold))
    print(f'Fold {fold+1}')
    train_dataset = [X_train[i] for i in train_index]
    val_dataset = [X_train[i] for i in val_index]

    # Set up Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=gpt2_classificaiton_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda p: {"accuracy": accuracy_score(p.predictions.argmax(-1), p.label_ids)},
    )
    # Train the model
    trainer.train()
    
    # Evaluate on test set per fold
    test_results = trainer.predict(X_test)
    logits = test_results.predictions
    # test_probs = torch.nn.functional.softmax(torch.from_numpy(logits), dim=-1).numpy() # TODO: aggregate by averaging over probs?
    test_predictions = logits.argmax(axis=1)

    # Aggregate the results from all folds
    all_test_labels.extend(y_test)
    all_test_predictions.extend(test_results.predictions.argmax(axis=1))

    # Evaluate on test set per fold
    accuracy_per_fold = accuracy_score(y_test_ids, test_predictions)
    all_fold_test_accuracies.append(accuracy_per_fold)

########################################################### Test Evaluation ##############################################################
# Double check the alignment between all_test_labels and all_test_predictions!
# print('all_test_labels: ', all_test_labels)
# print('all_test_predictions: ', all_test_predictions)

# Convert test_labels from string to id
all_test_labels_ids = []
for label_str in all_test_labels:
   label_id = labels_ids[label_str]
   all_test_labels_ids.append(label_id)

# Aggregate results across all folds
agg_accuracy = accuracy_score(all_test_labels_ids, all_test_predictions)
avg_accuracy = np.mean(all_fold_test_accuracies)
std_accuracy = np.std(all_fold_test_accuracies)
accuracy = {'avg_accuracy': avg_accuracy, 'std_accuracy': std_accuracy}

# print(f'Complete training over {k_folds} folds for seed {seed}, {num_epochs} epochs, lr ={learning_rate_str}, model {model_name_or_path}, {cognition_type}')
# print(f'Aggregated test accuracy: {agg_accuracy}')
# print(f'Average test accuracy: {avg_accuracy}')
# print(f'Standard deviation of test accuracy: {std_accuracy}')

############################################### Save Evaluation Resluts ##############################################################
# Specify the path to save the plots
if max_sequence_len is None:
  save_dir = f'./results/{model_name_or_path}/{cognition_type}/lr_{learning_rate_str}_bs_{batch_size}_epochs_{num_epochs}/seed_{seed}'
else:
  save_dir = f'./results/{model_name_or_path}/{cognition_type}/maxlen_{max_sequence_len}/lr_{learning_rate_str}_bs_{batch_size}_epochs_{num_epochs}/seed_{seed}'
os.makedirs(save_dir, exist_ok=True)

# Save the absolute classification report for averaging over seeds later
report_path = os.path.join(save_dir, 'classification_report.json')
with open(report_path, 'w') as f:
    report_dict = classification_report(all_test_labels_ids,
                                            all_test_predictions,
                                            labels=list(labels_ids.values()),
                                            target_names=list(labels_ids.keys()),
                                            output_dict=True)
    json.dump(report_dict, f)

# Save the confusion matrix in loadable formats for later averaging over seeds too
agg_confusion_matrix = plot_confusion_matrix(y_true=all_test_labels_ids,
                                            y_pred=all_test_predictions,
                                            classes=list(labels_ids.keys()),
                                            normalize=False, # don't normalize because we need the absolute numbers for averaging over seeds
                                            magnify=0.1)
cm_path = os.path.join(save_dir, 'confusion_matrix.pkl')
with open(cm_path, 'wb') as f:
    pickle.dump(agg_confusion_matrix, f)


# Save overall accuracy for the seed
json_output = os.path.join(save_dir, 'accuracy.json')
with open(json_output, 'w') as f:
    json.dump(accuracy, f)


