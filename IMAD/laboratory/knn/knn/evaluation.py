import warnings

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn import metrics as mtr
from sklearn import model_selection as ms
from sklearn.neighbors.classification import KNeighborsClassifier


warnings.filterwarnings('ignore', category=UndefinedMetricWarning)


def evaluate_knn_all_options(x, y, fold_nums=None):
    n_neighbors = list(range(1, 6))
    weights = ['uniform', 'distance', 'custom']
    metrics = ['euclidean', 'manhattan', 'chebyshev']

    fold_generators = [('KFold', ms.KFold),
                       ('StratifiedKFold', ms.StratifiedKFold)]

    results = {}

    for fg_name, fold_generator in fold_generators:
        print('Fold generator:', fg_name)
        results[fg_name] = {}

        print('For all neighbourhood sizes...')
        results[fg_name]['neighbour'] = _eval_knn_for_all_folds(
            x, y, fold_generator, fold_nums, 'n_neighbors', n_neighbors
        )

        print('For all weighting methods...')
        results[fg_name]['weight'] = _eval_knn_for_all_folds(
            x, y, fold_generator, fold_nums, 'weights', weights
        )

        print('For all distance metric types...')
        results[fg_name]['metric'] = _eval_knn_for_all_folds(
            x, y, fold_generator, fold_nums, 'metric', metrics
        )

    return results


def _eval_knn_for_all_folds(x, y,
                            fold_generator, fold_nums,
                            parameter_name, parameter_values):
    results = {}

    for p in parameter_values:
        rs = {
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1': []
        }

        if p == 'custom':
            options = {'weights': custom_weighing_method}
        else:
            options = {parameter_name: p}

        for fold_num in fold_nums:
            folds = fold_generator(n_splits=fold_num)
            r = evaluate_knn(options, x, y, folds)
            rs['Accuracy'].append(r['Accuracy'])
            rs['Precision'].append(r['Precision'])
            rs['Recall'].append(r['Recall'])
            rs['F1'].append(r['F1'])

        results[p] = rs

    return results


def evaluate_knn(options, x, y, folds):
    return evaluate_classifier(KNeighborsClassifier, options, x, y, folds)


def evaluate_classifier(classifier_cls, classifier_cls_kwargs, x, y, folds):
    result = {}

    # If it is binary classification
    scoring_avg_type = 'binary' if len(set(y)) == 2 else 'macro'

    scoring_methods = (('Accuracy', mtr.accuracy_score, {}),
                       ('Precision', mtr.precision_score,
                        {'average': scoring_avg_type}),
                       ('Recall', mtr.recall_score,
                        {'average': scoring_avg_type}),
                       ('F1', mtr.f1_score, {'average': scoring_avg_type}))

    classifier = classifier_cls(**classifier_cls_kwargs)
    y_pred = ms.cross_val_predict(classifier, x, y, cv=folds)

    for sm_name, sm, sm_kwargs in scoring_methods:
        score = sm(y, y_pred, **sm_kwargs)
        result[sm_name] = np.round(score, 3)

    return result


def custom_weighing_method(dist):
    return np.random.normal(size=dist.shape)