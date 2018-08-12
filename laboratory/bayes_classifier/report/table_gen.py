

input_data = [
{'Accuracy': [0.972, 0.961, 0.961, 0.955, 0.961, 0.961, 0.961, 0.955], 'Precision': [0.973, 0.961, 0.959, 0.954, 0.96, 0.959, 0.959, 0.954], 'Recall': [0.972, 0.964, 0.965, 0.96, 0.964, 0.965, 0.965, 0.96], 'F1': [0.972, 0.962, 0.962, 0.957, 0.962, 0.962, 0.962, 0.957]},
{'Accuracy': [0.899, 0.899, 0.904, 0.904, 0.893, 0.899, 0.904, 0.899], 'Precision': [0.902, 0.903, 0.908, 0.908, 0.897, 0.902, 0.907, 0.902], 'Recall': [0.91, 0.909, 0.914, 0.914, 0.904, 0.908, 0.913, 0.908], 'F1': [0.903, 0.904, 0.909, 0.909, 0.899, 0.904, 0.909, 0.904]},
{'Accuracy': [0.893, 0.888, 0.91, 0.899, 0.904, 0.91, 0.916, 0.916], 'Precision': [0.901, 0.896, 0.917, 0.906, 0.911, 0.916, 0.92, 0.92], 'Recall': [0.905, 0.898, 0.921, 0.908, 0.915, 0.92, 0.924, 0.924], 'F1': [0.9, 0.894, 0.915, 0.904, 0.91, 0.915, 0.92, 0.92]},
{'Accuracy': [0.916, 0.899, 0.904, 0.916, 0.899, 0.91, 0.904, 0.904], 'Precision': [0.924, 0.908, 0.911, 0.922, 0.906, 0.915, 0.91, 0.91], 'Recall': [0.926, 0.911, 0.917, 0.928, 0.91, 0.92, 0.915, 0.915], 'F1': [0.92, 0.905, 0.911, 0.921, 0.905, 0.915, 0.91, 0.91]}
]


def make_line_in_tex_table(data, index):
    res = " & ".join(map(lambda x: str(x), ["", index] + data[index]))
    return res + " \\\ \\cline{2-10} \n"


def make_table(data):
    fmt_str = """
\\begin{table}[H]
    \\center
        \\caption{}
        \\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|}
            \\hline
            \multirow{2}{*}{\\textbf{Metoda dyskr.}} & \multirow{2}{*}{\\textbf{Metryka}} & \multicolumn{8}{|c|}{\\textbf{CV}} \\\ \\cline{3-10}
                            &  & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 \\\ \\hline                
            %s
            
            \\hline
    \\end{tabular}        
\\end{table}
    """

    table_data = ""
    disc_methods = ['Brak', 'Equal-width', 'Equal-freq', 'CAIM']

    for idx, d in enumerate(data):

        table_data += '\multirow{4}{*}{\\textit{%s}} ' % disc_methods[idx]
        for key in d.keys():
            table_data += make_line_in_tex_table(d, key)
        table_data += '\n\n'

    return fmt_str % table_data


if __name__ == "__main__":
    print(make_table(input_data))
