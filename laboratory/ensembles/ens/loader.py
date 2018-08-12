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
