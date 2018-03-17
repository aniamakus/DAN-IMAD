import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import naive_bayes as nb

import bc.discretization as discr
from bc.evaluation import evaluate_classifier
import bc.loader as ldr
import bc.visualization as vis


def main():
    """
    1. LOAD DATASET
    - ldr.load_iris()
    - ldr.load_diabetes()
    - ldr.load_wine()
    - ldr.load_glass()

    Returns:
        (data, targets, attribute_names)
    """

    x, y, attr_names = ldr.load_diabetes()

    """
    2. DISCRETIZE (OR NOT)
    
    if yes:
        x = discr.discretize_data_wrapper(METHOD, x, y)
    else:
        # do nothing :)
        
    Methods:
    - discr.equal_freq
    - discr.equal_width
    - discr.caim_binning
    """
    x = discr.discretize_data_wrapper(discr.equal_freq, x, y)

    """
    3. CREATE CLASSIFIER
    
    if discretize:
        nbcls = nb.MultinomialNB
    else:
        nbcls = nb.GaussianNB
    """
    nbcls = nb.MultinomialNB

    """
    4. EVALUATE / SCORE
    
    Define cross-validation type and size...
    ...and then pass it into prepared function (evaluate_classifier)
    """
    cv_val = 5
    fold_gen = StratifiedKFold

    scores = evaluate_classifier(nbcls, x, y, fold_gen, cv_val,
                                 is_binary_classification=True)
    print(scores)

    """
    5. CONFUSION MATRIX
    
    Plot the confusion matrix (normed)
    """
    fig = vis.plot_confusion_matrix(scores['Confusion_Matrix'],
                                    class_names=['Healthy', 'Ill'],
                                    title='Confusion Matrix')
    plt.show(fig)


def make_confusion_matrix_graphs():
    params = [
        # Diabetes
        dict(
            class_names=['Healthy', 'Ill'],
            ds_loader=ldr.load_diabetes,
            ds_name='Diabetes',
            discr_method=None,
            cv_val=9,
            fold_gen=KFold,
        ),
        dict(
            class_names=['Healthy', 'Ill'],
            ds_loader=ldr.load_diabetes,
            ds_name='Diabetes',
            discr_method=None,
            cv_val=6,
            fold_gen=StratifiedKFold,
        ),

        # Wine
        dict(
            class_names=['1', '2', '3'],
            ds_loader=ldr.load_wine,
            ds_name='Wine',
            discr_method=None,
            cv_val=8,
            fold_gen=KFold,
        ),
        dict(
            class_names=['1', '2', '3'],
            ds_loader=ldr.load_wine,
            ds_name='Wine',
            discr_method=None,
            cv_val=2,
            fold_gen=StratifiedKFold,
        ),

        # Glass
        dict(
            class_names=['1', '2', '3', '5', '6', '7'],
            ds_loader=ldr.load_glass,
            ds_name='Glass',
            discr_method=discr.caim_binning,
            cv_val=9,
            fold_gen=KFold,
        ),
        dict(
            class_names=['1', '2', '3', '5', '6', '7'],
            ds_loader=ldr.load_glass,
            ds_name='Glass',
            discr_method=discr.caim_binning,
            cv_val=5,
            fold_gen=StratifiedKFold,
        ),
    ]

    for p in params:
        vis.generate_confusion_matrix_plot(
            filename_prefix='cm', should_save=True, **p
        )


def make_scoring_graphs():
    vis.generate_dataset_scoring_plots(
        'scoring_kfold',
        fold_gen=StratifiedKFold,
        should_save=False
    )


def make_discretization_graphs():
    vis.generate_dataset_attributes_distribution_plots(
        'ef',
        discr_method=discr.equal_freq,
        should_save=False
    )


if __name__ == "__main__":
    main()
