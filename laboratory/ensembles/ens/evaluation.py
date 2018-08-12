import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn import metrics as mtr
from sklearn import model_selection as ms

warnings.filterwarnings('ignore', category=UndefinedMetricWarning)


def evaluate_ensemble(ens_config, x, y, nb_folds):
    scoring_avg_type = 'binary' if len(set(y)) == 2 else 'macro'

    if not isinstance(nb_folds, list):
        nb_folds = list([nb_folds])

    f1s = []
    for cv in nb_folds:
        classifier = ens_config[3]()
        folds = ms.StratifiedKFold(n_splits=cv)
        y_pred = ms.cross_val_predict(classifier, x, y, cv=folds)

        f1 = mtr.f1_score(y, y_pred, average=scoring_avg_type)
        f1s.append(np.round(f1, 3))

    df = pd.DataFrame()
    df['cv'] = nb_folds
    df['f1'] = f1s
    df['ensemble_algorithm'] = ens_config[0]
    df['parameter_name'] = ens_config[1]
    df['parameter_value'] = ens_config[2]

    return df
