from copy import deepcopy

from sklearn import base
import numpy as np


class LogTransformMonetaryValues(base.BaseEstimator, base.TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X):
        return self

    def transform(self, X):
        X = deepcopy(X)

        for i in range(2, 289 + 1, 7):
            which_var = [f'X{i}', f'X{i+1}', f'X{i+2}', f'X{i+3}']
            # we add 1 to avoid problem of taking logarithm of 0
            X[which_var] = np.log10(X[which_var] + 1)

        return X


class CutoffHighIncome(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, cutoff_percentile=0.99):
        self.cutoff_percentile = cutoff_percentile

    def fit(self, y):
        self.cutoff_value = np.quantile(y, self.cutoff_percentile)

        return self

    def transform(self, y):
        y = deepcopy(y)

        y[y > self.cutoff_value] = self.cutoff_value

        return y


class AddGenderFeature(base.BaseEstimator, base.TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X):
        return self

    def transform(self, X):
        X = deepcopy(X)

        X['X296'] = (X['X50'] > 0) | (X['X162'] > 0) | (X['X260'])

        return X