"""Dataset loader"""
import pandas as pd


FILE_PATH_FMT = 'data/{}.data.txt'


def load_dataset(name):
    filepath = FILE_PATH_FMT.format(name)
    df = pd.read_csv(filepath, header=0)

    attribute_names = df.columns.values.tolist()
    attribute_names.remove('Class')
    x = df[attribute_names]
    y = df['Class']

    return x, y, attribute_names


# def load_dataset(name, cls_col=-1, data_col_begin=0, data_col_end=-1):
#     filepath = 'data/{}.data.txt'.format(name)
#
#     df = pd.read_csv(filepath, header=0)
#     dataset = df.get_values()
#     attribute_names = df.columns.values[data_col_begin:data_col_end]
#
#     data = dataset[:, data_col_begin:data_col_end]
#     target = dataset[:, cls_col]
#     return data.astype('float64'), target, attribute_names