"""
Ensembles tests
"""
import pandas as pd

from ens import configs as cfg
from ens import evaluation as eval
from ens import export as exp
from ens import loader as ldr

pd.set_option('display.width', 500)
pd.set_option('display.max_rows', 500)


def post_process_df(df):
    total_df = df.copy()
    total_df = total_df.reset_index()
    del total_df['index']

    total_df = total_df[['ds_name', 'ensemble_algorithm',
                         'parameter_name', 'parameter_value',
                         'cv', 'f1']]
    total_df['parameter_value'] = total_df['parameter_value'].astype(str)

    return total_df


def run_tests():
    dataset_names = ('diabetes', 'glass', 'wine')
    configs = cfg.make_all_configs()
    nb_folds = list(range(2, 10))
    total_df = pd.DataFrame()

    for ds_name in dataset_names:
        print(f'Dataset: {ds_name}')
        x, y, _ = ldr.load_dataset(ds_name)

        for ens_alg_name in configs.keys():
            print(f'Ensemble algortithm: {ens_alg_name}')

            for parameter_name in configs[ens_alg_name].keys():
                print(f'Parameter name: {parameter_name}')

                for config in configs[ens_alg_name][parameter_name]:
                    print(f'Config: {config}')

                    df = eval.evaluate_ensemble(config, x, y, nb_folds)
                    df['ds_name'] = ds_name

                    total_df = total_df.append(df)

    total_df = post_process_df(total_df)
    # print(to_pivot_table(total_df))

    dfs = exp.make_disjoint_dfs(total_df)
    exp.make_latex_tables(dfs)
    exp.make_plots(dfs)


if __name__ == "__main__":
    run_tests()
