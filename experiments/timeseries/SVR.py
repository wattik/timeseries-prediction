# coding=utf-8
import warnings

import matplotlib.pyplot as plt
from sklearn.svm import SVR

from evaluation.metrics import rae_scorer, mse_scorer
from evaluation.timeseries import TimeSeriesEvaluation
from evaluation.model import ModelDefinition, ReferenceModel
from transformations import *

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

"""
Data columns: idx | initial columns (1...n) | Volume 
"""

# Definitions
horizon = 2  # todo: plot covariance as a func of time-shift
volume_col = "Volume"
date_col = "Date"

# Transformations
feature_transformations = [
    Fork([
        Pipe([
            WeekDay(date_col),
            OneHot(),
        ]),
        Pipe([
            ColumnSelection([volume_col]),
            Percentile([10, 90]),
            OneHot(),
            TimeSeriesExpansion(1)
        ]),
        Pipe([
            ColumnSelection([volume_col]),
            ExponentialMovingMean(7, decay=0.9)
        ]),
        Pipe([
            ColumnSelection([volume_col]),
            GeometricMean(30)
        ]),
        Pipe([
            # ColumnDrop([date_col]),
            ColumnSelection([volume_col]),
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
]

transformations = DataTransformations(feature_transformations, target_transformations)

# Data
data = pd.read_csv("data/sp500.csv")
target, features = transformations.transform_target(data), transformations.transform_features(data)

# Learning
params = {
    "C": 1,
    "epsilon": 0.05,
    "kernel": "linear",
}
model_class = SVR
title = "SVR - %s, h=%d, window=%d" % (params["kernel"], horizon, 300)

model_definition = ModelDefinition(model_class, params)
time_series_evaluation = TimeSeriesEvaluation(300, 300, scorer=mse_scorer)

results = time_series_evaluation.eval(model_definition, target, features)
print(results)
plt.plot(results.reference_scores.scores, label="Reference")
plt.plot(results.model_scores.scores, label="Model")
plt.legend()
plt.title(title)
plt.text(0, min(results.reference_scores.scores),
         "R2 = %.4f +- %.4f" % (results.model_scores.mean, results.model_scores.std))
plt.ylim(min(results.reference_scores.scores), 1)
plt.show()

# # #593
model, X, y = time_series_evaluation.get_model(593, model_definition.get_model(), target, features)
y_pred = model.predict(X)
reference = ReferenceModel(target)

y_pred.columns=["Model"]
# y_pred["diff_model"] = abs(y-y_pred)
y_pred["Volume"] = y
y_pred["Reference"] = reference.predict(X)
# y_pred["diff_reference"] = abs(y-reference.predict(X))
y_pred.plot()

plt.show()
