import itertools

import numpy as np


def make_wpe_metric(percent):
    def func(y_true, y_pred):
        low_boundary = y_true * (1 - percent / 100)
        high_boundary = y_true * (1 + percent / 100)

        return (
            np.sum((low_boundary < y_pred) & (y_pred < high_boundary))
            / len(y_pred)
            * 100
        )

    return func


wpe_20 = make_wpe_metric(20.0)
wpe_10 = make_wpe_metric(10.0)


def proper_ordering(y_true, y_pred):
    proper = 0
    not_proper = 0
    for pair_a, pair_b in itertools.combinations(zip(y_true, y_pred), 2):
        if pair_a[0] < pair_b[0] and pair_a[1] < pair_b[1]:
            proper += 1
        elif pair_a[0] > pair_b[0] and pair_a[1] > pair_b[1]:
            proper += 1
        else:
            not_proper += 1

    return proper / (proper + not_proper) * 100


# oblicza jakie jest prawdopodobieństwo, że
# jeśli ktoś zarabia więcej niż 'threshold'
# to predykcja naszego modelu także będzie większa niż threshold
def one_tail_accuracy(y_true, y_pred, threshold):
    idx = y_true >= threshold

    return np.sum(y_pred[idx] >= threshold) / np.sum(idx) * 100


def classification_accuracy(y_true, y_pred, threshold):
    idx_upward = y_true >= threshold
    idx_downward = y_true < threshold

    return (
        (
            np.sum(y_pred[idx_upward] >= threshold)
            + np.sum(y_pred[idx_downward] < threshold)
        )
        / len(y_true)
        * 100
    )
