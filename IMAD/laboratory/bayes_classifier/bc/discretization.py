"""Discretization methods"""
from copy import deepcopy

import numpy as np
from caimcaim import CAIMD


def discretize_data(data, method, **kwargs):
    data_cpy = deepcopy(data)

    _, nb_cols = data_cpy.shape

    if method is caim_binning:
        x_discr = caim_binning(data, **kwargs)
        for idx in range(nb_cols):
            data_cpy[:, idx] = x_discr[:, idx]
    else:
        for idx in range(nb_cols):
            data_cpy[:, idx] = method(data_cpy[:, idx], **kwargs)

    return data_cpy


def equal_width(attr_data, nb_bins):
    attr_data = attr_data.astype('float64')
    _, bins = np.histogram(attr_data, bins=nb_bins)
    print(bins)
    return np.fmin(np.digitize(attr_data, bins), nb_bins)


def equal_freq(attr_data, nb_bins):
    """Inspired by:
    https://stackoverflow.com/questions/39418380/histogram-with-equal-number-of-points-in-each-bin
    """
    attr_data = attr_data.astype('float64')

    nb_data = len(attr_data)
    bins = np.interp(np.linspace(0, nb_data, nb_bins + 1),
                     np.arange(nb_data),
                     np.sort(attr_data))
    print(bins)
    return np.fmin(np.digitize(attr_data, bins), nb_bins)


def caim_binning(x, y):
    """https://github.com/airysen/caimcaim"""
    caim = CAIMD()
    x_discr = caim.fit_transform(x, y)
    return x_discr


# def main():
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import seaborn as sns
#     from sklearn.datasets import load_iris
#
#     from collections import Counter
#
#     fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
#
#     iris = load_iris()
#     x = iris.data
#     y = iris.target
#
#     print('Before:')
#     print(x[:, 0])
#     sns.distplot(x[:, 0], ax=ax1)
#
#     x_binned_ew = equal_width(x[:, 0], 3)
#     print('Equal-width:')
#     print(x_binned_ew)
#     print(Counter(x_binned_ew))
#     sns.distplot(x_binned_ew, ax=ax2)
#
#     x_binned_ef = equal_freq(x[:, 0], 3)
#     print('Equal-freq:')
#     print(x_binned_ef)
#     print(Counter(x_binned_ef))
#     sns.distplot(x_binned_ef, ax=ax3)
#
#     x_binned_caim = caim_binning(x, y)
#     x_binned_caim = x_binned_caim[:, 0]
#     print('CAIM:')
#     print(x_binned_caim)
#     print(Counter(x_binned_caim))
#     sns.distplot(x_binned_caim, ax=ax4)
#
#     plt.show()
#
#
# if __name__ == "__main__":
#     main()
