# coding=utf-8
from typing import Tuple, Type, NoReturn, Union

from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from data_primitives import Target, Features, Matrix
import pandas as pd


class Normalizer:
    def __init__(self):
        self.features_norm = StandardScaler()
        self.target_norm = StandardScaler()

    def fit(self, X: Matrix, y: Matrix) -> NoReturn:
        self.features_norm.fit(X)
        self.target_norm.fit(y)

    def transform(self, X: Matrix, y: Matrix = None) -> Union[Tuple[Matrix, Matrix], Matrix]:
        if y is None:
            return self.features_norm.transform(X)
        else:
            return self.features_norm.transform(X), self.target_norm.transform(y).ravel()

    def reverse(self, y: Matrix) -> Matrix:
        return self.target_norm.inverse_transform(y)


class Model(BaseEstimator):
    def __init__(self, model_cls: Type[BaseEstimator], params):
        self.norm = Normalizer()
        self.model = model_cls(**params)

        self.target_name = None
        self.column_names = None

    def fit(self, features: Features, target: Target):
        self.target_name = target.columns[0]
        self.column_names = list(features.columns.values)

        self.norm.fit(features.values, target.values)
        X, y = self.norm.transform(features.values, target.values)
        self.model.fit(X, y)

    def predict(self, features: Features) -> Target:
        X = self.norm.transform(features.values)
        y = self.model.predict(X)
        return self._make_target(self.norm.reverse(y), features.index)

    def get_parameters(self, deep=True):
        return {
            "model_params": self.model.get_params(deep),
            "model_attributes": self._get_model_attributes()
        }

    def _get_model_attributes(self):
        return {
            "feature_column_names": self.column_names,
        }

    def _make_target(self, array, index=None) -> Target:
        df = pd.DataFrame(array, index=index, columns=[self.target_name])
        return Target(df)


class ReferenceModel:
    def __init__(self, dataset_target: Target):
        self.predictions = dataset_target.shift().fillna(method="backfill")

    def predict(self, features: Features) -> Target:
        return self.predictions.loc[features.index]


class ModelDefinition:
    def __init__(self, model_cls: Type[BaseEstimator], params):
        self.model_cls = model_cls
        self.params = params

    def get_model(self) -> Model:
        return Model(self.model_cls, self.params)
