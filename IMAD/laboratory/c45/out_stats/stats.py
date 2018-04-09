"""
Generate stats of attributes
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    datasets = [
        # '../data/iris.data.txt',
        '../data/glass2.data.txt',
        '../data/wine.data.txt',
        '../data/diabetes.data.txt',
    ]
    
    for ds in datasets:
        df = pd.read_csv(ds, header=0)

        attr_names = list(df.columns.values)
        attr_names.remove('Class')
        print(attr_names)

        analyzed_df = pd.DataFrame(columns=('Name', 'Min', 'Max', 
                                            'Mean', 'Std',
                                            'Distribution'))

        for an in attr_names:
            vals = df[an]

            plt.figure()
            sns.distplot(vals, kde=False,
                         hist_kws=dict(alpha=1, rwidth=0.75), 
                         bins=30)
            plt.xlabel("")
            plt.ylabel("")
            plt.savefig(an + '.png')

            analyzed_df = analyzed_df.append({
                'Name': an,
                'Min': np.round(min(vals), 2),
                'Max': np.round(max(vals), 2),
                'Mean': np.round(np.mean(vals), 2),
                'Std': np.round(np.std(vals), 2),
                'Distribution': '{img/stats/%s.png}' % an
            }, ignore_index=True)

        with open(ds[8:], 'w') as tbl_file:
            tbl_file.write(analyzed_df.to_latex(index=False))


if __name__ == "__main__":
    main()

