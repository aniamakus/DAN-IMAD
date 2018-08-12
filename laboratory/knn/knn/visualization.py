"""All plots related functions"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def make_scoring_plots(dataset_name, results):
    sns.set()
    sns.set_palette(sns.color_palette('hls', 8))

    for fold_generator in results.keys():
        for parameter in results[fold_generator].keys():
            r = results[fold_generator][parameter]

            fig = plt.figure(figsize=(10, 5))
            axs = fig.subplots(1, 4, sharey=True)

            plt.suptitle(parameter)
            plt.setp(axs, xticks=list(range(2, 10)))

            for parameter_val in r.keys():
                for idx, metric in enumerate(r[parameter_val].keys()):
                    ax = axs[idx]
                    ax.plot(list(range(2, 10)), r[parameter_val][metric],
                            label=parameter_val, linestyle='--', marker='o')
                    ax.set_title(metric)
                    # ax.set_ylim((0, 1))

            fig.legend(r.keys(), loc='upper center', bbox_to_anchor=(0.5, 0.05),
                       fancybox=True, shadow=True, ncol=len(r.keys()))
            fig.tight_layout()
    plt.show()


            # filepath = 'out_plots/{}_{}_{}.png'.format(dataset_name,
            #                                            fold_generator,
            #                                            parameter)
            #
            # plt.savefig(filepath)
            # print('Saved to:', filepath)
            # plt.close()


def make_scoring_barplots(dataset_name, results):
    sns.set()
    sns.set_palette(sns.color_palette('hls', 8))

    for fold_generator in results.keys():
        for parameter in results[fold_generator].keys():
            r = results[fold_generator][parameter]

            fig = plt.figure(figsize=(10, 5))
            axs = fig.subplots(1, 4, sharey=True)

            plt.suptitle(parameter)

            agg = {
                'Accuracy': [],
                'Precision': [],
                'Recall': [],
                'F1': []
            }
            for parameter_val in r.keys():
                for idx, metric in enumerate(r[parameter_val].keys()):
                    agg[metric].append(r[parameter_val][metric][0])

            for idx, metric in enumerate(agg.keys()):
                ax = axs[idx]
                g = sns.barplot(list(r.keys()), agg[metric],
                                label=parameter, ax=ax)
                # for idx2, v in enumerate(r.keys()):
                #     g.text(idx2, agg[metric][idx2], agg[metric][idx2],
                #            color='black', ha="center")
                ax.set_title(metric)
                ax.set_ylim((0, 1))

            fig.tight_layout()

    plt.show()
            # filepath = 'out_plots/{}_{}_{}.png'.format(dataset_name,
            #                                            fold_generator,
            #                                            parameter)
            #
            # plt.savefig(filepath)
            # print('Saved to:', filepath)
            # plt.close()
