# coding=utf-8

import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

from evaluation.model import ModelDefinition
from evaluation.timeseries import TimeSeriesEvaluation
from transformations import *

"""
Data columns: idx | initial columns (1...n) | Volume 
"""

# Definitions
horizon = 6
volume_col = "Volume"
date_col = "Date"

# Transformations
feature_transformations = [
    Fork([
        Pipe([
            WeekDay(date_col),
            OneHot()
        ]),
        Pipe([
            ColumnSelection([volume_col]),
            Percentile([5, 10, 20, 80, 90, 95]),
            OneHot(),
            TimeSeriesExpansion(1)
        ]),
        Pipe([
            # ColumnDrop([date_col]),
            ColumnSelection([volume_col]),
            Multiply(1.0 / 1000000),
            # Fork([Identity(), PolynomialLiftup(2), PolynomialLiftup(3)]),
            TimeSeriesExpansion(horizon),
        ])
    ]),
    CutNTop(horizon)
]

target_transformations = [
    ColumnSelection([volume_col]),
    CutNTop(horizon)
]

transformations = DataTransformations(feature_transformations, target_transformations)

# Data
data = pd.read_csv("data/sp500.csv")
target, features = transformations.transform_target(data), transformations.transform_features(data)

# Learning
params = {
    "solver": "adam",
    "activation": "tanh",
    "hidden_layer_sizes": tuple(20 for _ in range(5)),
    "max_iter": 10000,
}
model_class = MLPRegressor
model_definition = ModelDefinition(model_class, params)

time_series_evaluation = TimeSeriesEvaluation(300, 300)
results = time_series_evaluation.eval(model_definition, target, features)

print(results)
plt.plot(results.reference_scores.results, label="Reference")
plt.plot(results.model_scores.results, label="Model")
plt.legend()
plt.show()
