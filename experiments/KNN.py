# coding=utf-8
from pprint import pprint

from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor

from evaluation.model import Model
from transformations import *
import pandas as pd

from evaluation.drawing import BarsDrawer

"""
Data columns: idx | initial columns (1...n) | Volume 
"""

# Definitions
horizon = 10  # todo: plot covariance as a func of time-shift
volume_col = "Volume"

# Transformations
feature_transformations = [
    # ColumnDrop(["Date"]),
    ColumnSelection([volume_col]),
    # PolynomialLiftup(2),
    TimeSeriesExpansion(horizon),
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
    "n_neighbors": 6
}

model = Model(KNeighborsRegressor, params)
model.fit(*training_data)
prediction = model.predict(testing_data.features)

score = r2_score(testing_data.target, prediction)

print(score)

BarsDrawer(testing_data.target, prediction).show()

pprint(model.get_parameters())


