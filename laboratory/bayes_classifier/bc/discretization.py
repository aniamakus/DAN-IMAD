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
    return np.fmin(np.digitize(attr_data, bins), nb_bins)


def caim_binning(x, y):
    """https://github.com/airysen/caimcaim"""
    caim = CAIMD()
    x_discr = caim.fit_transform(x, y)
    return x_discr


def calculate_freedman_diaconis(x):
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    nb_bins = int((2 * iqr) / np.cbrt(x.shape[0]))
    #print('Freedman–Diaconis:', nb_bins)
    return nb_bins


def discretize_data_wrapper(discr_method, x, y):
    if discr_method in (equal_freq, equal_width):
        kwargs = dict(nb_bins=calculate_freedman_diaconis(x))
    else:
        kwargs = dict(y=y)
    x = discretize_data(x, discr_method, **kwargs)
    return x
