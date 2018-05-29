"""
Experiment configs module
"""
from sklearn import ensemble
from sklearn import tree


def get_general_config():
    base_estimator = tree.DecisionTreeClassifier()
    n_estimators = [10, 25, 50, 75, 99]

    ensemble_algorithms = {
        'adaboost': {
            'class': lambda kws: ensemble.AdaBoostClassifier(
                base_estimator=base_estimator, **kws),
            'params': {
                'n_estimators': n_estimators,
                'learning_rate': [0.0001, 0.001, 0.01, 0.1],
                'algorithm': ['SAMME', 'SAMME.R']
            }
        },
        'bagging': {
            'class': lambda kws: ensemble.BaggingClassifier(
                base_estimator=base_estimator, **kws),
            'params': {
                'n_estimators': n_estimators,
                'max_samples': [0.25, 0.5, 0.75, 1.0],
                'bootstrap': [True, False]
            }
        },
        'random-forest': {
            'class': lambda kws: ensemble.RandomForestClassifier(**kws),
            'params': {
                'n_estimators': n_estimators,
                'criterion': ['gini', 'entropy'],
                'bootstrap': [True, False],
                'max_features': [0.25, 0.5, 0.75, 1.0]
            }
        }
    }

    return ensemble_algorithms


def make_all_configs():
    ensemble_algorithms = get_general_config()
    all_configs = {}

    for ens_name, ens_config in ensemble_algorithms.items():
        all_configs[ens_name] = {}
        for p_name, p_values in ens_config['params'].items():
            all_configs[ens_name][p_name] = []
            for value in p_values:

                ens_class = ens_config['class']
                ens_kwargs = {
                    p_name: value
                }
                ens = lambda: ens_class(ens_kwargs)

                all_configs[ens_name][p_name].append(
                    (ens_name, p_name, value, ens)
                )

    return all_configs
