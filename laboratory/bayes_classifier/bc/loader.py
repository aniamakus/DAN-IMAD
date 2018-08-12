"""Dataset loader"""
import pandas as pd


def load_dataset(name, cls_col=-1, data_col_begin=0, data_col_end=-1):
    filepath = 'data/{}.data.txt'.format(name)

    df = pd.read_csv(filepath, header=0)
    dataset = df.get_values()
    attribute_names = df.columns.values[data_col_begin:data_col_end]

    data = dataset[:, data_col_begin:data_col_end]
    target = dataset[:, cls_col]
    return data.astype('float64'), target, attribute_names


def load_iris():
    return load_dataset('iris',
                        data_col_begin=0,
                        data_col_end=-1,
                        cls_col=-1)


def load_glass():
    return load_dataset('glass',
                        data_col_begin=1,
                        data_col_end=10,
                        cls_col=-1)


def load_diabetes():
    return load_dataset('pima-indians-diabetes',
                        data_col_begin=0,
                        data_col_end=8,
                        cls_col=-1)


def load_wine():
    return load_dataset('wine',
                        data_col_begin=1,
                        data_col_end=14,
                        cls_col=0)
