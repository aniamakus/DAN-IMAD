"""All plots related functions"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics as mtr
from sklearn.model_selection import cross_val_predict
from sklearn import naive_bayes as nb
from sklearn.exceptions import UndefinedMetricWarning

from bc import discretization as disc
from bc import loader


warnings.filterwarnings('ignore', category=UndefinedMetricWarning)


def generate_dataset_attributes_distribution_plots(filename_prefix,
                                                   discr_method=None,
                                                   should_save=True):
    dataset_loaders = (loader.load_wine,
                       loader.load_diabetes,
                       loader.load_glass)
    sns.set()
    sns.set_palette(sns.color_palette('hls', 8))

    for ds_idx, dsl in enumerate(dataset_loaders):
        print('Loading:', dsl.__name__)
        x, y, attrs = dsl()

        if discr_method is not None:
            x = disc.discretize_data_wrapper(discr_method, x, y)

        fig = plt.figure(ds_idx, figsize=(10, 5))

        for attr_idx, attr in enumerate(attrs):
            print('Plotting attr:', attr)
            ax = plt.subplot(int(len(attrs)/4)+1, 4, attr_idx + 1)
            g = sns.distplot(x[:, attr_idx], kde=False,
                             hist_kws=dict(alpha=1, rwidth=0.75),
                             bins=30, ax=ax)
            g.set(title=attr)

        fig.tight_layout()

        if should_save:
            filepath = 'plots_out/{}_{}.png'.format(filename_prefix,
                                                    dsl.__name__)
            plt.savefig(filepath)
            print('Saved to:', filepath)

    if not should_save:
        plt.show()


def generate_dataset_scoring_plots(filename_prefix,
                                   cv_min=2, cv_max=9,
                                   fold_gen=None,
                                   should_save=True):
    if fold_gen is None:
        raise ValueError('Please provide fold generator')

    dataset_loaders = (loader.load_wine,
                       loader.load_diabetes,
                       loader.load_glass)
    discretization_methods = ((None, 'No discretization'),
                              (disc.equal_width, 'Equal-width'),
                              (disc.equal_freq, 'Equal-frequency'),
                              (disc.caim_binning, 'CAIM'))
    scoring_methods = ((mtr.accuracy_score, 'Accuracy'),
                       (mtr.precision_score, 'Precision'),
                       (mtr.recall_score, 'Recall'),
                       (mtr.f1_score, 'F1'))

    sns.set()
    sns.set_palette(sns.color_palette('hls', 8))

    for ds_idx, dsl in enumerate(dataset_loaders):
        fig = plt.figure(ds_idx, figsize=(10, 5))
        axs = fig.subplots(1, 4, sharey=True)
        plt.setp(axs, xticks=list(range(cv_min, cv_max + 1)))

        for dm, dm_name in discretization_methods:
            print('DM:', dm_name)

            print('Loading:', dsl.__name__)
            x, y, attrs = dsl()

            if dm:
                nbcls = nb.MultinomialNB
                x = disc.discretize_data_wrapper(dm, x, y)
            else:
                nbcls = nb.GaussianNB

            results = {
                'Accuracy': [],
                'Precision': [],
                'Recall': [],
                'F1': []
            }
            cvs = range(cv_min, cv_max + 1)

            for cv in cvs:
                y_pred = cross_val_predict(nbcls(), x, y,
                                           cv=fold_gen(n_splits=cv))

                for sm, sm_name in scoring_methods:
                    if sm_name == 'Accuracy':
                        score = sm(y, y_pred)
                    else:
                        if dsl.__name__ == 'load_diabetes':
                            scoring_avg_type = 'binary'
                        else:
                            scoring_avg_type = 'macro'

                        score = sm(y, y_pred, average=scoring_avg_type)

                    results[sm_name].append(np.round(score, 3))

            for sm_idx, (_, sm_name) in enumerate(scoring_methods):
                ax = axs[sm_idx]
                ax.plot(list(cvs), results[sm_name],
                        label=dm_name if sm_idx == 0 else "")
                ax.set_title(sm_name)

            print(results)

        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05),
                   fancybox=True, shadow=True, ncol=4)
        fig.tight_layout()

        if should_save:
            filepath = 'plots_out/{}_{}.png'.format(filename_prefix,
                                                    dsl.__name__)
            plt.savefig(filepath)
            print('Saved to:', filepath)

    if not should_save:
        plt.show()


def generate_confusion_matrix_plot(filename_prefix,
                                   class_names=None,
                                   ds_loader=None,
                                   ds_name=None,
                                   discr_method=None,
                                   cv_val=None,
                                   fold_gen=None,
                                   should_save=True):
    x, y, attrs = ds_loader()
    if discr_method:
        nbcls = nb.MultinomialNB
        x = disc.discretize_data_wrapper(discr_method, x, y)
    else:
        nbcls = nb.GaussianNB

    y_pred = cross_val_predict(nbcls(), x, y, cv=fold_gen(n_splits=cv_val))
    cm = mtr.confusion_matrix(y, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    title = 'Confusion Matrix ({}; cv={}; {})'.format(ds_name,
                                                      cv_val,
                                                      fold_gen.__name__)

    fig = plot_confusion_matrix(cm, class_names, title)

    if not should_save:
        plt.show(fig)
    else:
        filepath = 'plots_out/{}_{}_cv{}_{}.png'.format(
            filename_prefix,
            ds_name,
            cv_val,
            fold_gen.__name__
        )
        fig.savefig(filepath)
        print('Saved to:', filepath)


def plot_confusion_matrix(cm, class_names, title):
    fig = plt.figure()
    ax = fig.subplots(1, 1)
    g = sns.heatmap(cm, annot=True,
                    xticklabels=class_names,
                    yticklabels=class_names,
                    cmap=sns.color_palette("coolwarm", 7),
                    ax=ax)

    g.set(title=title,
          xlabel='Predicted label',
          ylabel='True label')

    return fig



