# coding=utf-8
import pandas as pd
import numpy as np
from evaluation.model import Model


def analyze_weights(model: Model, columns):
    zero_sample = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

    offset = model.predict(zero_sample).values.ravel()
    weights = pd.DataFrame(offset, columns=["offset"])

    for column in columns:
        zero_sample[column] = 1.0
        weights[column] = model.predict(zero_sample)
        weights[column] -= weights["offset"]
        zero_sample[column] = 0.0

    return weights

