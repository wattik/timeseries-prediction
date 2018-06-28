# coding=utf-8
import numpy as np

from sklearn.metrics import mean_absolute_error, r2_score


def relative_absolute_error(y, y_pred):
    return mean_absolute_error(y, y_pred) / np.mean(y)


def relative_absolute_error_per_sample(y, y_pred):
    return np.mean(np.abs(y_pred - y) / y)


def relative_absolute_error_vec(y, y_pred):
    return np.abs(y_pred - y) / y


def mean_squared_error(y, y_pred):
    diff = (y - y_pred)
    std = diff.std()
    return np.sqrt(((diff / std) ** 2).mean()) * std


# Scorers:

def r2_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return r2_score(y, y_pred)


def mse_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return mean_squared_error(y, y_pred)


def rae_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return relative_absolute_error(y, y_pred)
