from typing import NewType, NamedTuple

import numpy as np
import pandas as pd

"""A properly set up DataFrame with corresponding columns, indexed sequentially."""
SourceData = NewType("SourceData", pd.DataFrame)

"""DataFrames containing the numerical representation of data, learning-ready."""
Features = NewType("Features", pd.DataFrame)
Target = NewType("Target", pd.DataFrame)

TrainingData = NamedTuple('TrainingData', features=Features, target=Target)
TestingData = NamedTuple('TestingData', features=Features, target=Target)

Matrix = NewType("Matrix", np.ndarray)

"""Numpy's ndarrays carrying the indices of training and testing data splits."""
Indices = NewType("Indices", np.ndarray)
Split = NamedTuple("Split", train_idx=Indices, test_idx=Indices)
