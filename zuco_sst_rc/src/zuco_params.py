import argparse
import random
import torch
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name_or_path',
                        type=str,
                        default='bert-base-uncased')
    parser.add_argument('--output_dir',
                        type=str,
                        default='/../content/results/')
    parser.add_argument('--run_name',
                        type=str,
                        default='/../content/results/')
    parser.add_argument('--task',
                        type=str,
                        default='sst',
                        help="sst, rel")
    parser.add_argument('--report_to',
                        type=str,
                        default='none',
                        help="none, wandb, azure_ml, comet_ml, mlflow, tensorboard, all")
    parser.add_argument('--zuco',
                        type=bool,
                        default=None)
    parser.add_argument('--binary',
                        type=bool,
                        default=None,
                        help="SST-3 with neutral removed.")
    parser.add_argument('--direct',
                        type=bool,
                        default=None)
    parser.add_argument('--pred_zuco',
                        type=bool,
                        default=None)
    parser.add_argument('--zuco_splits',
                        type=bool,
                        default=None)
    parser.add_argument('--use_wandb',
                        type=bool,
                        default=None)
    parser.add_argument('--cv',
                        type=int,
                        default=0,
                        help="Stratified k-fold cross-validation.")
    parser.add_argument('--save_total_limit',
                        type=int,
                        default=None,
                        help="Max checkpoints.")
    parser.add_argument('--use_weights',
                        type=bool,
                        default=None,
                        help="Use attention weights vs. raw scores.")
    parser.add_argument('--model_att',
                        type=bool,
                        default=None)
    parser.add_argument('--att_only',
                        type=bool,
                        default=None,
                        help="Freeze all but attention weights, train on ZuCo.")
    parser.add_argument('--zuco_only',
                        type=bool,
                        default=None,
                        help="Use ZuCo dataset for joint loss.")
    parser.add_argument('--pred_only',
                        type=bool,
                        default=None,
                        help="ZuCo value prediction loss only.")
    parser.add_argument('--study_name',
                        type=str,
                        default='zuco')
    parser.add_argument('--save',
                        type=bool,
                        default=False),
    parser.add_argument('--save_steps',
                        type=int,
                        default=80),
    parser.add_argument('--load_best',
                        type=bool,
                        default=None),
    parser.add_argument('--save_att',
                        type=bool,
                        default=None),
    parser.add_argument('--per_sample',
                        type=bool,
                        default=None),
    parser.add_argument('--aug',
                        type=bool,
                        default=None),
    parser.add_argument('--save_preds',
                        type=bool,
                        default=None),
    parser.add_argument('--train_type',
                        type=str,
                        default='train',
                        help='train, grid')
    parser.add_argument('--et_type',
                        type=str,
                        default='trt',
                        help='ffd, mfd, trt')
    parser.add_argument('--pred_type',
                        type=str,
                        default='eeg',
                        help='eeg, et')
    parser.add_argument('--chkpt',
                        type=str,
                        default='2000')
    parser.add_argument('--eval_strat',
                        type=str,
                        default='epoch',
                        help='steps, epoch, no')
    parser.add_argument('--overwrite_output_dir',
                        type=bool,
                        default=False)
    parser.add_argument('--filtered',
                        type=bool,
                        default=None,
                        help='For baseline.')
    parser.add_argument('--freeze',
                        type=bool,
                        default=None,
                        help='Freeze all but attention weights.')
    parser.add_argument('--batch_size',
                        dest='batch_size',
                        type=int,
                        default=32)
    parser.add_argument('--eeg_lmbda',
                        type=float,
                        default=0)
    parser.add_argument('--et_lmbda',
                        type=float,
                        default=0)
    parser.add_argument('--pred_lmbda',
                        type=float,
                        default=0.5)
    parser.add_argument('--random_scores',
                        type=bool,
                        default=None)
    parser.add_argument('--layer_handling',
                         type=str,
                         default='-1',
                         help='avg, every, l (specific layer)')
    parser.add_argument('--head_handling',
                         type=str,
                         default='avg',
                         help='z (specific heads), avg, max, every')
    parser.add_argument('--electrode_handling',
                         type=str,
                         default='avg',
                         help='max, avg, sum')
    parser.add_argument('--et_head',
                        type=int,
                        default=0)
    parser.add_argument('--eeg_head',
                        type=int,
                        default=1)
    parser.add_argument('--et_layer',
                        type=int,
                        default=0)
    parser.add_argument('--eeg_layer',
                        type=int,
                        default=11)
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--eval_steps', type=int, default=25,
                        help='Use with eval_strat=steps.')
    parser.add_argument('--do_train', type=bool, default=None)
    parser.add_argument('--do_eval', type=bool, default=None)
    parser.add_argument('--do_predict', type=bool, default=None)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        torch.cuda.manual_seed_all(args.seed)
    except:
        pass

    return args

args = parse_args()
