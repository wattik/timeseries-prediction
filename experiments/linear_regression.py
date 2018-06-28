# coding=utf-8
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

from evaluation.metrics import relative_absolute_error_per_sample
from evaluation.model import Model
from transformations import *
import pandas as pd

from evaluation.drawing import BarsDrawer

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
    # Normalize()
]

transformations = DataTransformations(feature_transformations, target_transformations)

splitter = Splitter(ratio=0.4, seed=42)
dataset_maker = DatasetMaker(transformations, splitter)

# Data
data = pd.read_csv("data/sp500.csv")
training_data, testing_data = dataset_maker.make(data)

# Learning
params = {
    "fit_intercept": False,
}

model = Model(LinearRegression, params)
model.fit(*training_data)
prediction = model.predict(testing_data.features)

BarsDrawer(testing_data.target, prediction).show()

print("Coefficients:", model.model.coef_)
pprint(model.get_parameters())

print("R2:", r2_score(testing_data.target, prediction))
print("mean abs error:", mean_absolute_error(testing_data.target, prediction))
print("% mean abs error:", relative_absolute_error_per_sample(testing_data.target, prediction))

columns = testing_data.features.columns
zero_sample = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

offset = model.predict(zero_sample).values.ravel()
weights = pd.DataFrame(offset, columns=["offset"])

for column in columns:
    zero_sample[column] = 1.0
    weights[column] = model.predict(zero_sample)
    weights[column] -= weights["offset"]
    zero_sample[column] = 0.0

print(weights.to_string())

print(pd.DataFrame([data[volume_col].mean()], columns=["Average"]))