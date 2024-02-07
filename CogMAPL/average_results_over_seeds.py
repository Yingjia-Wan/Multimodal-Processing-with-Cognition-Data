import numpy as np
import json
import os
import pickle
from sklearn.metrics import classification_report
from ml_things import plot_dict
import matplotlib.pyplot as plt
import argparse
import time

######################################################### Plot Hyperparameters ################################
# Magnify intervals where font size matters
MAGNIFY_INTERVALS = [0.1, 1]

# Min and max appropriate font sizes
FONT_RANGE = [10.5, 50]

# Maximum allowed magnify. This will get multiplied by 0 - 1 value.
MAX_MAGNIFY = 15

# Increase font for title ratio.
TITLE_FONT_RATIO = 1.1

########################################################### Hyperparameter Set up ###########################################################
def parse_args():
  # Create parser
  parser = argparse.ArgumentParser(description='Hyperparameters for model training')
  # Add arguments
  parser.add_argument('--model_name_or_path', type=str, required=True)
  parser.add_argument('--batch_size', type=int, required=False)
  parser.add_argument('--learning_rate', type=float, required=False)
  parser.add_argument('--num_epochs', type=int, required=False)
  parser.add_argument('--cognition_type', type=str, required=True)
  parser.add_argument('--search', type=bool, required=False, default=False)
  # Parse arguments
  args = parser.parse_args()
  return args


labels_ids = {'negative': 0, 'neutral':1, 'positive': 2}

args = parse_args()
model_name_or_path = args.model_name_or_path # 'gpt2' or 'gpt2-medium' or 'gpt2-large' or 'gpt2-xl' or 'EleutherAI/gpt-neo-125M' or 'EleutherAI/gpt-neo-1.3B' or 'EleutherAI/gpt-neo-2.7B'
batch_size = args.batch_size
learning_rate = args.learning_rate # best so far is 3e-5 for ET
learning_rate_str = "{:.0e}".format(learning_rate)
num_epochs = args.num_epochs
cognition_type = args.cognition_type
search = args.search

seeds = [16, 17, 18, 19, 20]


if search:
    from training_kfold_decoder_search import best_params
    print('Average over best_params:', best_params)
    model_name_or_path = best_params['model_name_or_path']
    batch_size = best_params['batch_size']
    learning_rate = best_params['learning_rate']
    learning_rate_str = "{:.0e}".format(learning_rate)
    num_epochs = best_params['num_epochs']

result_dir = f'./results/{model_name_or_path}/{cognition_type}/lr_{learning_rate_str}_bs_{batch_size}_epochs_{num_epochs}'

######################################################### caculating ###########################################


def average_results_over_seeds(result_dir, seeds):
    # Initialize sums for confusion matrix & classification report
    sum_cm = None
    sum_report = None
    sum_macro_avg = {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}
    sum_weighted_avg = {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}
    sum_accuracy = 0

    for seed in seeds:
        # Load the confusion matrix & classification report for this seed
        cm_path = os.path.join(result_dir, f'seed_{seed}', 'confusion_matrix.pkl')
        with open(cm_path, 'rb') as f:
            cm = pickle.load(f)

        report_path = os.path.join(result_dir, f'seed_{seed}', 'classification_report.json')
        with open(report_path, 'r') as f:
            report = json.load(f)

        # Convert the report to a confusion matrix-like format for easier averaging
        report_cm = np.array([[v['precision'], v['recall'], v['f1-score'], v['support']] for k, v in report.items() if k in ['negative', 'neutral', 'positive']])
        macro_avg = report['macro avg']
        weighted_avg = report['weighted avg']
        accuracy = report['accuracy']

        # Add this seed's confusion matrix and classification report to the sum
        if sum_cm is None:
            sum_cm = cm
            sum_report = report_cm
        else:
            sum_cm += cm
            sum_report += report_cm

        for key in sum_macro_avg.keys():
            sum_macro_avg[key] += macro_avg[key]
            sum_weighted_avg[key] += weighted_avg[key]
        sum_accuracy += accuracy
        time.sleep(2)  # Introduce a 1-second delay between file operations to avoid overloading and file_not_found errors

    # Divide by the number of seeds to get the average
    avg_cm = sum_cm / len(seeds)
    avg_report = sum_report / len(seeds)
    avg_macro_avg = {key: val / len(seeds) for key, val in sum_macro_avg.items()}
    avg_weighted_avg = {key: val / len(seeds) for key, val in sum_weighted_avg.items()}
    avg_accuracy = sum_accuracy / len(seeds)

    # Convert the average classification report back to the original format
    avg_report_dict = {
        'negative': {'precision': avg_report[0, 0], 'recall': avg_report[0, 1], 'f1-score': avg_report[0, 2], 'support': avg_report[0, 3]},
        'neutral': {'precision': avg_report[1, 0], 'recall': avg_report[1, 1], 'f1-score': avg_report[1, 2], 'support': avg_report[1, 3]},
        'positive': {'precision': avg_report[2, 0], 'recall': avg_report[2, 1], 'f1-score': avg_report[2, 2], 'support': avg_report[2, 3]}
    }
    avg_report_dict['macro avg'] = avg_macro_avg
    avg_report_dict['weighted avg'] = avg_weighted_avg
    avg_report_dict['accuracy'] = avg_accuracy

    return avg_cm, avg_report_dict



def average_accuracy_over_seeds(result_dir, seeds):
    sum_accuracy = 0
    sum_std = 0
    seeds_with_error = []
    for seed in seeds:
        accuracy_path = os.path.join(result_dir, f'seed_{seed}', 'accuracy.json')
        # Load the accuracy file
        try:
            with open(accuracy_path, 'r') as f:
                accuracy = json.load(f)
            sum_accuracy += accuracy['avg_accuracy']
            sum_std += accuracy['std_accuracy']
        except:
            print(f'Error loading accuracy file for seed {seed}!!!')
            seeds_with_error.append(seed)
            continue
        time.sleep(2)  # Introduce a delay between file operations to avoid overloading and file_not_found errors
    avg_accuracy = sum_accuracy / (len(seeds) - len(seeds_with_error))
    avg_std = sum_std / (len(seeds) - len(seeds_with_error))
    print(f'Number of error seeds: {len(seeds_with_error)}','\n', f'Seeds with loading error: {seeds_with_error}')
    return avg_accuracy, avg_std


def main():
    avg_cm, avg_report = average_results_over_seeds(result_dir, seeds)
    # Save the results
    with open(os.path.join(result_dir, 'average_confusion_matrix.pkl'), 'wb') as f:
        pickle.dump(avg_cm, f)
    avg_cm_plot_dir = os.path.join(result_dir, 'average_confusion_matrix.png')
    my_plot_cm(avg_cm, use_title='Normalized Confusion Matrix (averaged over seeds)', classes=list(labels_ids.keys()), normalize=True, font_size = 20, path = avg_cm_plot_dir)

    with open(os.path.join(result_dir, 'average_classification_report.json'), 'w') as f:
        json.dump(avg_report, f, indent=4)

    avg_accuracy, avg_std = average_accuracy_over_seeds(result_dir, seeds)
    with open(os.path.join(result_dir, 'average_accuracy.json'), 'w') as f:
        json.dump({'accuracy': avg_accuracy, 'std': avg_std}, f, indent=4)

    # Print or save the results
    print(f'All done! Averaged over seeds, the avg and std acc for {cognition_type}, {model_name_or_path}, {learning_rate_str}, {num_epochs} are:')
    print('Accuracy: {:.4f} ± {:.4f}'.format(avg_accuracy, avg_std))
    print('--------------------------------------------------')
    return


def my_plot_cm(cm, use_title=None, classes='', normalize=True, style_sheet='ggplot',
                          cmap=plt.cm.Blues, font_size=None, verbose=0, width=3, height=1, magnify=0.1, use_dpi=50,
                          path=None, show_plot=True, **kwargs):
    r"""
    This modified function prints and plots the confusion matrix, by Yingjia Wan

    Normalization can be applied by setting `normalize=True`.

    Arguments:

        cm (:obj:`pkl`):
            ararys of confusion matrix.

        use_title (:obj:`int`, `optional`):
            Title on top of plot. This argument is optional and it will have a `None` value attributed
            inside the function.

        classes (:obj:`str`, `optional`, defaults to :obj:``):
            List of label names. This argument is optional and it has a default value attributed
            inside the function.

        normalize (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Normalize confusion matrix or not. This argument is optional and it has a default value attributed
            inside the function.

        style_sheet (:obj:`str`, `optional`, defaults to :obj:`ggplot`):
            Style of plot. Use plt.style.available to show all styles. This argument is optional and it has a default
            value attributed inside the function.

        cmap (:obj:`str`, `optional`, defaults to :obj:`plt.cm.Blues`):
            It is a plt.cm plot theme. Plot themes: `plt.cm.Blues`, `plt.cm.BuPu`, `plt.cm.GnBu`, `plt.cm.Greens`,
            `plt.cm.OrRd`. This argument is optional and it has a default value attributed inside the function.

        font_size (:obj:`int` or `float`, `optional`):
            Font size to use across the plot. By default this function will adjust font size depending on `magnify`
            value. If this value is set, it will ignore the `magnify` recommended font size. The title font size is by
            default `1.8` greater than font-size. This argument is optional and it will have a `None` value attributed
            inside the function.

        verbose (:obj:`int`, `optional`, defaults to :obj:`0`):
            To display confusion matrix value or not if set > 0. This argument is optional and it has a default
            value attributed inside the function.

        width (:obj:`int`, `optional`, defaults to :obj:`3`):
            Horizontal length of plot. This argument is optional and it has a default value attributed inside
            the function.

        height (:obj:`int`, `optional`, defaults to :obj:`1`):
            Height length of plot in inches. This argument is optional and it has a default value attributed inside
            the function.

        magnify (:obj:`float`, `optional`, defaults to :obj:`0.1`):
            Ratio increase of both with and height keeping the same ratio size. This argument is optional and it has a
            default value attributed inside the function.

        use_dpi (:obj:`int`, `optional`, defaults to :obj:`50`):
            Print resolution is measured in dots per inch (or “DPI”). This argument is optional and it has a default
            value attributed inside the function.

        path (:obj:`str`, `optional`):
            Path and file name of plot saved as image. If want to save in current path just pass in the file name.
            This argument is optional and it will have a None value attributed inside the function.

        show_plot (:obj:`bool`, `optional`, defaults to :obj:`1`):
            if you want to call `plt.show()`. or not (if you run on a headless server). This argument is optional and
            it has a default value attributed inside the function.

        kwargs (:obj:`dict`, `optional`):
            Other arguments that might be deprecated or not included as details. This argument is optional and it will
            have a `None` value attributed inside the function.

    Returns:

        :obj:`np.ndarray`: Confusion matrix used to plot.

    Raises:

        DeprecationWarning: If arguments `title` is used.

        DeprecationWarning: If arguments `image` is used.

        DeprecationWarning: If arguments `dpi` is used.

        ValueError: If `y_true` and `y_pred` arrays don't have same length.

        ValueError: If `dict_arrays` doesn't have string keys.

        ValueError: If `dict_arrays` doesn't have array values.

        ValueError: If `style_sheet` is not valid.

        DeprecationWarning: If `magnify` is se to values that don't belong to [0, 1] values.

        ValueError: If `font_size` is not `None` and smaller or equal to 0.

    """

    # Handle deprecation warnings if `title` is used.
    if 'title' in kwargs:
        # assign same value
        use_title = kwargs['title']
        warnings.warn("`title` will be deprecated in future updates. Use `use_title` in stead!", DeprecationWarning)

    # Handle deprecation warnings if `image` is used.
    if 'image' in kwargs:
        # assign same value
        path = kwargs['image']
        warnings.warn("`image` will be deprecated in future updates. Use `path` in stead!", DeprecationWarning)
    # Handle deprecation warnings if `dpi` is used.
    if 'dpi' in kwargs:
        # assign same value
        use_dpi = kwargs['dpi']
        warnings.warn("`dpi` will be deprecated in future updates. Use `use_dpi` in stead!", DeprecationWarning)


    # Make sure style sheet is correct.
    if style_sheet in plt.style.available:
        # set style of plot
        plt.style.use(style_sheet)
    else:
        # style is not correct
        raise ValueError("`style_sheet=%s` is not in the supported styles: %s" % (str(style_sheet),
                                                                                  str(plt.style.available)))

    # Make sure `magnify` is in right range.
    if magnify > 1 or magnify <= 0:
        # Deprecation warning from last time.
        warnings.warn(f'`magnify` needs to have value in [0,1]! `{magnify}` will be converted to `0.1` as default.',
                      DeprecationWarning)
        # Convert to regular value 0.1.
        magnify = 0.1

    # Make sure `font_size` is set right.
    if (font_size is not None) and (font_size <= 0):
        # Raise value error -  is not correct.
        raise ValueError(f'`font_size` needs to be positive number! Invalid value {font_size}')

    # Font size select custom or adjusted on `magnify` value.
    font_size = font_size if font_size is not None else np.interp(magnify, MAGNIFY_INTERVALS, FONT_RANGE)

    # Font variables dictionary. Keep it in this format for future updates.
    font_dict = dict(
        family='DejaVu Sans',
        color='black',
        weight='normal',
        size=font_size,
    )

    # Normalize setup.
    if normalize is True:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        use_title = 'Normalized confusion matrix' if use_title is None else use_title
    else:
        use_title = 'Confusion matrix, without normalization' if use_title is None else use_title

    # Print if verbose.
    print(cm) if verbose > 0 else None

    # Plot setup.
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # Show all ticks.
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # Label ticks with the respective list entries.
           xticklabels=classes, yticklabels=classes,
           )

    # Set horizontal axis name.
    ax.set_xlabel('Predicted label', fontdict=font_dict)

    # Set vertical axis name.
    ax.set_ylabel('True label', fontdict=font_dict)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.grid(False)

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontdict=font_dict)

    # Adjust both axis labels font size at same time.
    plt.tick_params(labelsize=font_dict['size'])

    # Adjust font for title.
    font_dict['size'] *= TITLE_FONT_RATIO

    # Set title of figure.
    plt.title(use_title, fontdict=font_dict)

    # Rescale `magnify` to be used on inches.
    magnify *= MAX_MAGNIFY

    # Never display grid.
    plt.grid(False)

    # Make figure nice.
    plt.tight_layout()

    # Get figure object from plot.
    fig = plt.gcf()

    # Get size of figure.
    figsize = fig.get_size_inches()

    # Change size depending on height and width variables.
    figsize = [figsize[0] * width * magnify, figsize[1] * height * magnify]

    # Set the new figure size with magnify.
    fig.set_size_inches(figsize)

    # There is an error when DPI and plot size are too large!
    try:
        # Save figure to image if path is set.
        fig.savefig(path, dpi=use_dpi, bbox_inches='tight') if path is not None else None
    except ValueError:
        # Deprecation warning from last time.
        warnings.warn(f'`magnify={magnify // 15}` is to big in combination'
                      f' with `use_dpi={use_dpi}`! Try using lower values for'
                      f' `magnify` and/or `use_dpi`. Image was saved in {path}'
                      f' with `use_dpi=50 and `magnify={magnify // 15}`!', Warning)
        # Set DPI to smaller value and warn user to use smaller magnify or smaller dpi.
        use_dpi = 50
        # Save figure to image if path is set.
        fig.savefig(path, dpi=use_dpi, bbox_inches='tight') if path is not None else None

    # Show plot.
    plt.show() if show_plot is True else None

    return cm



main()