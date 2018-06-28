# coding=utf-8

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from evaluation.model import ModelDefinition
from evaluation.timeseries import TimeSeriesEvaluation
from transformations import *

"""
Data columns: idx | initial columns (1...n) | Volume 
"""

# Definitions
horizon = 6  # todo: plot covariance as a func of time-shift
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
            Percentile([10, 90]),
            OneHot(),
            TimeSeriesExpansion(1)
        ]),
        Pipe([
            ColumnSelection([volume_col]),
            ExponentialMovingMean(horizon, decay=0.1)
        ]),
        Pipe([
            ColumnSelection([volume_col]),
            GeometricMean(20)
        ]),
        Pipe([
            # ColumnDrop([date_col]),
            ColumnSelection([volume_col]),
            Multiply(1.0 / 1e6),
            # Fork([Identity(), PolynomialLiftup(2)]),
            TimeSeriesExpansion(horizon),
            # Normalize()
        ])
    ]),
    CutNTop(horizon),
    # Normalize()
]

target_transformations = [
    ColumnSelection([volume_col]),
    CutNTop(horizon),
    # Normalize()
]

transformations = DataTransformations(feature_transformations, target_transformations)

# Data
data = pd.read_csv("data/sp500.csv")
target, features = transformations.transform_target(data), transformations.transform_features(data)

# Learning
params = {
    "fit_intercept": False,
}

model_class = LinearRegression
title = "Linear Regression"

model_definition = ModelDefinition(model_class, params)
time_series_evaluation = TimeSeriesEvaluation(300, 300)
results = time_series_evaluation.eval(model_definition, target, features)

print(results)
plt.plot(results.reference_scores.results, label="Reference")
plt.plot(results.model_scores.results, label="Model")
plt.legend()
plt.title(title)
plt.text(0, min(results.reference_scores.results),
         "R2 = %.4f +- %.4f" % (results.model_scores.mean, results.model_scores.std))
plt.ylim(min(results.reference_scores.results), 1)
plt.show()
