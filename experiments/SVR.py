# coding=utf-8
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.svm import SVR

from evaluation.inspection import analyze_weights
from evaluation.kfolding import KFoldEvaluator
from evaluation.metrics import relative_absolute_error_per_sample, mse_scorer, rae_scorer
from evaluation.model import Model, ModelDefinition, ReferenceModel
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
    # Normalize()
]

transformations = DataTransformations(feature_transformations, target_transformations)

splitter = Splitter(ratio=0.2, seed=50, k=5)
dataset_maker = DatasetMaker(transformations, splitter)

# Data
data = pd.read_csv("data/sp500.csv")
data["Volume"] = data["Volume"]
# Learning
params = {
    "C": 1,
    "epsilon": 0.02,
    "kernel": "linear",
}
model_class = SVR

# Evaluation
training_data, testing_data = dataset_maker.make(data)

model = Model(model_class, params)
model.fit(*training_data)
prediction = model.predict(testing_data.features)

# Drawing
BarsDrawer(testing_data.target, prediction).show()

plt.figure()
scatter = pd.concat([testing_data.target["Volume"], prediction["Volume"]], axis=1)
scatter.columns = ["Volume", "Prediction"]

plt.ylim(testing_data.target["Volume"].min(), testing_data.target["Volume"].max())
plt.xlim(testing_data.target["Volume"].min(), testing_data.target["Volume"].max())
plt.grid(b="on")

sns.regplot(x="Volume", y="Prediction", data=scatter)
plt.show()

###

reference = ReferenceModel(dataset_maker.transformer.transform_target(data))

plt.figure()
scatter = pd.concat([testing_data.target["Volume"], reference.predict(testing_data.features)], axis=1)
scatter.columns = ["Volume", "Reference"]

plt.ylim(testing_data.target["Volume"].min(), testing_data.target["Volume"].max())
plt.xlim(testing_data.target["Volume"].min(), testing_data.target["Volume"].max())
plt.grid(b="on")

sns.regplot(x="Volume", y="Reference", data=scatter)
plt.show()

###

plt.figure()
sns.pairplot(testing_data.features[["Volume (T-1)_ema", "gma", "Volume (T-1)", "Volume (T-2)"]], diag_kind="kde", kind="reg")
plt.show()

# Scores
pprint(model.get_parameters())

print("R2:", r2_score(testing_data.target, prediction))
print("mean abs error:", mean_absolute_error(testing_data.target, prediction))
print("% mean abs error:", relative_absolute_error_per_sample(testing_data.target, prediction))

# Weights inspection
weights = analyze_weights(model, testing_data.features.columns)
print(weights.to_string())

# K Fold evaluation
k_folds = KFoldEvaluator(ModelDefinition(model_class, params), dataset_maker, scorer=rae_scorer)
results = k_folds.results(data)
print(results)
