"""
Tools for exporting results to files
"""


def results_to_file(dataset_name, results):
    tables = _results_to_table(results)

    for (fold_generator, parameter), table in tables.items():
        filepath = 'out_files/{}_{}_{}.txt'.format(dataset_name,
                                                   fold_generator,
                                                   parameter)
        with open(filepath, 'w') as f:
            f.write('\n'.join(table))


def results_to_latex(dataset_name, results):
    fmt_str = """
\\begin{table}[H]
    \\begin{tabular}{c|c|cccccccc}
       \multirow{2}{*}{Wartość parametru} & \multirow{2}{*}{Metryka} & \multicolumn{8}{|c|}{Kroswalidacja} \\\\
         & & K = 2 & K = 3 & K = 4 & K = 5 & K = 6 & K = 7 & K = 8 & K = 9 \\\\ \\hline
         %s \\\\ \\hline
    \end{tabular}
\end{table}
    """

    tables = _results_to_table(results)

    for (fold_generator, parameter), table in tables.items():
        filepath = 'out_files/{}_{}_{}.tex'.format(dataset_name,
                                                   fold_generator,
                                                   parameter)
        with open(filepath, 'w') as f:
            content = []
            for row in table:
                content.append(row.replace('[', '') \
                                  .replace(']', '') \
                                  .replace(',', '&'))

            content = '\\\\ \\hline\n'.join(content)
            f.write(fmt_str % content)


def _results_to_table(results):
    tables = {}

    for fold_generator in results.keys():
        for parameter in results[fold_generator].keys():
            r = results[fold_generator][parameter]

            lines = []
            for parameter_val in r.keys():
                for metric in r[parameter_val].keys():
                    lines.append(
                        '{},{},{}'.format(parameter_val, metric,
                                          r[parameter_val][metric])
                    )

            tables[(fold_generator, parameter)] = lines
    return tables
