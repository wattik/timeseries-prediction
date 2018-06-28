# coding=utf-8
from abc import ABC, abstractmethod

from numpy import prod
from typing import Tuple, List, Text

from data_primitives import SourceData, TrainingData, TestingData, Features, Target
from splitting import Splitter
from utils import apply

from itertools import combinations_with_replacement

import pandas as pd
import numpy as np


class ImmutableTransformation(ABC):
    @abstractmethod
    def __call__(self, df: SourceData) -> SourceData:
        pass


class ColumnSelection(ImmutableTransformation):
    def __init__(self, column_names: List[Text]):
        self.column_names = column_names

    def __call__(self, df: SourceData) -> SourceData:
        return df[self.column_names]


class ColumnDrop(ImmutableTransformation):
    def __init__(self, column_names: List[Text]):
        self.column_names = column_names

    def __call__(self, df: SourceData) -> SourceData:
        return SourceData(df.drop(self.column_names, axis=1))


class CutNTop(ImmutableTransformation):
    def __init__(self, n_rows: int):
        self.n_rows = n_rows

    def __call__(self, df: SourceData) -> SourceData:
        return df.iloc[self.n_rows:]


class TimeSeriesExpansion(ImmutableTransformation):
    def __init__(self, horizon: int):
        self.horizon = horizon

    def __call__(self, df: SourceData) -> SourceData:
        dfs = []
        for i in range(1, self.horizon + 1):
            term = df.shift(i)
            term = term.rename(columns=lambda s: str(s) + " (T-" + str(i) + ")")
            dfs.append(term)

        df = pd.concat(dfs, axis=1, join="inner")
        return df


class PolynomialLiftup(ImmutableTransformation):
    def __init__(self, order: int):
        assert order >= 2
        self.order = order

    def __call__(self, df: SourceData) -> SourceData:
        columns = list(df.columns)

        dfs = []
        for columns_comb in combinations_with_replacement(columns, self.order):
            columns_selection = list(columns_comb)
            selected_data = df[columns_selection]
            term = selected_data.aggregate(prod, axis=1)
            term.name = "*".join(columns_selection)
            dfs.append(term)

        return pd.concat(dfs, axis=1)


class Identity(ImmutableTransformation):
    def __call__(self, df: SourceData) -> SourceData:
        return df.copy()


class Multiply(ImmutableTransformation):
    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, df: SourceData) -> SourceData:
        return df * self.factor


class DateTimeAggr(ImmutableTransformation):
    def __init__(self, date_column_name: str, period, strftime=None):
        self.strftime = strftime
        self.period = period
        self.date_column_name = date_column_name

    def __call__(self, df: SourceData) -> SourceData:
        df = df.copy()
        datetime = pd.DatetimeIndex(df[self.date_column_name]).to_period(self.period)

        if self.strftime is not None:
            datetime = datetime.strftime(self.strftime)

        df[self.date_column_name] = datetime
        return df[self.date_column_name]


class WeekDay(ImmutableTransformation):
    def __init__(self, date_column_name: str):
        self.date_column_name = date_column_name

    def __call__(self, df: SourceData) -> SourceData:
        df = df.copy()
        datetime = pd.DatetimeIndex(df[self.date_column_name])
        datetime = datetime.weekday
        df[self.date_column_name] = datetime
        return df[self.date_column_name]


class OneHot(ImmutableTransformation):
    def __call__(self, df: SourceData) -> SourceData:
        onehot = pd.get_dummies(df)
        return onehot


class Percentile(ImmutableTransformation):
    def __init__(self, percentiles: List[int]):
        if percentiles[0] != 0:
            percentiles = [0] + percentiles
        self.percentiles = percentiles

    def __call__(self, df: SourceData) -> SourceData:
        percentile_values = np.percentile(df, self.percentiles)
        percentilized = np.digitize(df, percentile_values)
        percentilized_df = pd.DataFrame(percentilized, index=df.index, columns=[df.columns[0] + "-percentiles"])
        return percentilized_df.astype("category")


class Normalize(ImmutableTransformation):
    def __call__(self, df: SourceData) -> SourceData:
        return (df - df.mean()) / df.std()


class MovingAverage(ImmutableTransformation):
    def __init__(self, horizon):
        self.horizon = horizon
        self.pipe = TimeSeriesExpansion(horizon)

    def __call__(self, df: SourceData) -> SourceData:
        return self.pipe(df).mean(axis=1).rename(columns=lambda c: c + "_ama")


class ExponentialMovingMean(ImmutableTransformation):
    def __init__(self, horizon, decay: float = 0.5):
        self.decay = decay
        self.horizon = horizon
        self.pipe = TimeSeriesExpansion(horizon)

    def __call__(self, df: SourceData) -> SourceData:
        ema = self.pipe(df).iloc[:, ::-1].transpose().ewm(alpha=1-self.decay).mean().transpose().rename(
            columns=lambda c: c + "_ema")
        return ema.iloc[:, -1]


class GeometricMean(ImmutableTransformation):
    def __init__(self, horizon):
        self.horizon = horizon
        self.pipe = TimeSeriesExpansion(horizon)

    def __call__(self, df: SourceData) -> SourceData:
        gm = np.exp(np.log(self.pipe(df)).mean(axis=1))
        gm.name = "gma"
        return gm


class Fork(ImmutableTransformation):
    def __init__(self, teeth: List[ImmutableTransformation]):
        self.teeth = teeth

    def __call__(self, df: SourceData) -> SourceData:
        dfs = [f(df) for f in self.teeth]
        return pd.concat(dfs, axis=1, join="inner")


class Pipe(ImmutableTransformation):
    def __init__(self, elements: List[ImmutableTransformation]):
        self.elements = elements

    def __call__(self, df: SourceData) -> SourceData:
        return apply(self.elements, df)


class DataTransformations:
    def __init__(self, feature_transformations: List[ImmutableTransformation],
                 target_transformations: List[ImmutableTransformation]):
        self.target_transformations = target_transformations[:]
        self.feature_transformations = feature_transformations[:]

    def transform_features(self, df: SourceData) -> Features:
        return apply(self.feature_transformations, df)

    def transform_target(self, df: SourceData) -> Target:
        return apply(self.target_transformations, df)


class DatasetMaker:
    def __init__(self, data_transformations: DataTransformations, splitter: Splitter):
        self.splitter = splitter
        self.transformer = data_transformations

    def make(self, source_data: SourceData) -> Tuple[TrainingData, TestingData]:
        features = self.transformer.transform_features(source_data)
        target = self.transformer.transform_target(source_data)

        assert len(features) == len(target)
        split = self.splitter.train_test_indices(features)

        training_data = TrainingData(features.loc[split.train_idx], target.loc[split.train_idx])
        testing_data = TestingData(features.loc[split.test_idx], target.loc[split.test_idx])

        return training_data, testing_data

    def k_fold_make(self, source_data: SourceData) -> List[Tuple[TrainingData, TestingData]]:
        features = self.transformer.transform_features(source_data)
        target = self.transformer.transform_target(source_data)

        splits = self.splitter.k_fold_indices(features)

        folds = []
        for split in splits:
            training_data = TrainingData(features.loc[split.train_idx], target.loc[split.train_idx])
            testing_data = TestingData(features.loc[split.test_idx], target.loc[split.test_idx])
            folds.append((training_data, testing_data))

        return folds
