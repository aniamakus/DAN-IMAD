"""
Tools for exporting results to files
"""
import matplotlib.pyplot as plt
import pandas as pd


def to_pivot_table(df):
    df = df.copy()
    df = df.rename(columns={'ds_name': 'Zbiór danych',
                            'ensemble_algorithm': 'Algorytm',
                            'parameter_name': 'Parametr',
                            'parameter_value': 'Wartość parametru',
                            'cv': 'Liczba foldów',
                            'f1': 'Miara F1'})

    df = pd.pivot_table(df,
                        index=['Parametr', 'Wartość parametru'],
                        columns='Liczba foldów')
    return df


def make_latex_tables(dfs):
    for df in dfs:
        filename = f'out_files/{df[0]}.tex'
        pivot_df = to_pivot_table(df[1])

        print(f'Saving {filename}')
        print(pivot_df)
        print('\n\n')

        with open(filename, 'w') as f:
            table = pivot_df.to_latex()
            table = table.replace('toprule', 'hline')
            table = table.replace('midrule', 'hline')
            table = table.replace('bottomrule', 'hline')
            f.write(table)


def make_plots(dfs):
    for df in dfs:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1,
                                       sharex=True,
                                       figsize=(15, 10))

        # Grouped by cross-validation
        _make_plot_using_pivot(df, ax1)

        # Plot of max over all parameter values and all cross-validations
        _make_max_plot(df, ax2)

        ds_name, alg, p_name = df[0].split('_', maxsplit=2)
        fig.suptitle(
            f'Zbiór: {ds_name} Algorytm: {alg.upper()} Parametr: {p_name}'
        )

        fig.tight_layout()
        print(f'Saving plot {df[0]}')
        fig.savefig(f'out_plots/{df[0]}.png')
        plt.close()


def _make_max_plot(df, ax):
    actual_df = df[1]

    parameter_values = sorted(set(actual_df['parameter_value']))
    max_f1s = []
    for pv in parameter_values:
        max_f1s.append(
            max(actual_df[actual_df.parameter_value == pv]['f1'])
        )

    ax.plot(parameter_values, max_f1s, linestyle='--', marker='o')

    ax.set_xlabel('Wartość parametru')
    ax.set_ylabel('Wartość miary F1')


def _make_plot_using_pivot(df, ax):
    pivot_df = to_pivot_table(df[1])

    pivot_df.plot(kind='bar', ax=ax)

    ax.set_ylim((0, 1))

    ax.legend_.remove()
    ax.legend(list(range(2, 10)), loc='upper right',
              title='Liczba foldów', bbox_to_anchor=(1, 0.5))

    ax.set_xlabel('Wartość parametru')
    ax.set_ylabel('Wartość miary F1')


def make_disjoint_dfs(df):
    dfs = []

    for ds_name in set(df['ds_name']):
        ds_df = df[df.ds_name == ds_name]

        for ens_alg in set(ds_df['ensemble_algorithm']):
            ens_df = ds_df[ds_df.ensemble_algorithm == ens_alg]

            for p_name in set(ens_df['parameter_name']):
                p_df = ens_df[ens_df.parameter_name == p_name]

                filename = f'{ds_name}_{ens_alg}_{p_name}'

                dfs.append((filename, p_df))

    return dfs
