import numpy as np

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

from bc.discretization import discretize_data, equal_width, \
    equal_freq, caim_binning


def evaluate_classifier(x, y, discretize=None, cv_min=2, cv_max=10):
    result = {
        'accuracy': [],
        'precision_macro': [],
        'recall_macro': [],
        'f1_macro': [],
        'cnf_matrix': None,
    }

    if not discretize:
        bcls = GaussianNB
    else:
        bcls = MultinomialNB
        if discretize[0] == 'ew':  # Equal-with
            x = discretize_data(x, equal_width, nb_bins=discretize[1])
        elif discretize[0] == 'ef':  # Equal-frequency
            x = discretize_data(x, equal_freq, nb_bins=discretize[1])
        elif discretize[0] == 'caim':  # CAIM
            x = discretize_data(x, caim_binning, y=y)

    for method in ('accuracy', 'precision_macro', 'recall_macro', 'f1_macro'):
        for cv_val in range(cv_min, cv_max + 1):
            scores = cross_val_score(bcls(), x, y, scoring=method, cv=cv_val)
            result[method].append(scores.mean())

    bc = bcls()
    bc.fit(x, y)
    y_pred = bc.predict(x)
    cnf_matrix = confusion_matrix(y, y_pred)

    cm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    result['cnf_matrix'] = cm

    print(cm)

    return result
