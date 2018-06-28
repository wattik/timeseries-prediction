# coding=utf-8
from pprint import pprint

from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

from evaluation.model import Model
from transformations import *
import pandas as pd

from evaluation.drawing import BarsDrawer

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

splitter = Splitter(ratio=0.2)
dataset_maker = DatasetMaker(transformations, splitter)

# Data
data = pd.read_csv("data/sp500.csv")
training_data, testing_data = dataset_maker.make(data)

# Learning
params = {
    "solver": "adam",
    "activation": "tanh",
    "hidden_layer_sizes": tuple(20 for _ in range(5)),
    "max_iter": 10000,
}

model = Model(MLPRegressor, params)
model.fit(*training_data)
prediction = model.predict(testing_data.features)

pprint(model.get_parameters())

BarsDrawer(testing_data.target, prediction).show()
print("R2:", r2_score(testing_data.target, prediction))
