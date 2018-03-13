import numpy as np
from sklearn import metrics as mtr
from sklearn.model_selection import cross_val_predict


def evaluate_classifier(nbcls, x, y, fold_gen, cv_val,
                        is_binary_classification=False):
    result = {}

    scoring_methods = ((mtr.accuracy_score, 'Accuracy'),
                       (mtr.precision_score, 'Precision'),
                       (mtr.recall_score, 'Recall'),
                       (mtr.f1_score, 'F1'))

    y_pred = cross_val_predict(nbcls(), x, y,
                               cv=fold_gen(n_splits=cv_val))

    for sm, sm_name in scoring_methods:
        if sm_name == 'Accuracy':
            score = sm(y, y_pred)
        else:
            if is_binary_classification:
                scoring_avg_type = 'binary'
            else:
                scoring_avg_type = 'macro'

            score = sm(y, y_pred, average=scoring_avg_type)

        result[sm_name] = np.round(score, 3)

    cm = mtr.confusion_matrix(y, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    result['Confusion_Matrix'] = cm

    return result
