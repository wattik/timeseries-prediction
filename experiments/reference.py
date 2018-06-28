# coding=utf-8
from sklearn.metrics import r2_score
from sklearn.svm import SVR

from evaluation.drawing import BarsDrawer
from transformations import *
import pandas as pd

"""
Data columns: idx | initial columns (1...n) | Volume 
"""

# Definitions
horizon = 1
volume_col = "Volume"

# Transformations
feature_transformations = [
    ColumnSelection([volume_col]),
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
training_samples, testing_samples = dataset_maker.make(data)

# Evaluation
score = r2_score(testing_samples.target, testing_samples.features)
print(score)
BarsDrawer(testing_samples.target, testing_samples.features).show()
