"""
For generating report templates
"""


def main():
    datasets = ['Diabetes', 'Glass', 'Wine']
    algorithms = {
        'Adaboost': ['algorithm', 'learning_rate', 'n_estimators'],
        'Bagging': ['bootstrap', 'max_samples', 'n_estimators'],
        'Random-forest': ['bootstrap', 'criterion', 'n_estimators'],
    }

    for ds in datasets:
        print(f'Dataset: {ds}')

        with open(f'report/chapter_gen/results_{ds.lower()}.tex', 'w') as f:

            f.write('\\section{Zbiór %s}\n' % ds)

            for alg_name, params in algorithms.items():
                print(f'Algorithm: {alg_name}')
                f.write('\\subsection{Algorytm %s}\n' % alg_name)

                for param_name in params:
                    print(f'Parameter: {param_name}')

                    f_name = f'{ds.lower()}_{alg_name.lower()}_{param_name}'
                    caption = f'wartości miary F1 dla zbioru \"{ds}\" ' \
                              f'algorytmu \"{alg_name}\" przy ustalonym ' \
                              f'parametrze \"{param_name}\".'

                    f.write('\\input{resources/files/%s.tex}\n' % f_name)

                    f.write("""
\\begin{figure}[H]
    \\center
    \\includegraphics[width=\\textwidth]{resources/plots/%s.png}
    \\caption{Wykres %s}   
\\end{figure}                    
                    """ % (f_name, caption))
                    f.write('\n')


if __name__ == '__main__':
    main()