# coding=utf-8
from typing import Tuple

import numpy as np

from evaluation.metrics import r2_scorer
from data_primitives import Features, Target, Indices
from evaluation.model import ModelDefinition, ReferenceModel
from evaluation.scores import Scores, Results


class TimeSeriesEvaluation:
    def __init__(self, training_window: int, testing_window: int, scorer=r2_scorer):
        self.scorer = scorer
        self.testing_window = testing_window
        self.training_window = training_window

    def eval(self, model_factory: ModelDefinition, target: Target, features: Features) -> Results:
        index_min = target.index.min()
        index_max = target.index.max()

        reference = ReferenceModel(target)

        model_scores = []
        reference_scores = []

        for i, current_index in enumerate(
                list(range(index_min + self.training_window, index_max + 1 - self.testing_window))):
            if i % 50 == 0:
                print("Computing index: %d" % current_index)

            model = model_factory.get_model()
            model_score, reference_score = self.eval_at_time(current_index, model, reference, target, features)

            model_scores.append(model_score)
            reference_scores.append(reference_score)

        model_results = Scores(np.array(model_scores))
        reference_results = Scores(np.array(reference_scores))

        return Results(reference_results, model_results)

    def eval_at_time(self, time_id, model, reference, target: Target, features: Features) -> Tuple[float, float]:
        training_idx: Indices = list(range(time_id - self.training_window, time_id))
        testing_idx: Indices = list(range(time_id, time_id + self.testing_window + 1))

        model.fit(features.loc[training_idx], target.loc[training_idx])
        # `reference` is fitted already

        model_score = self.scorer(model, features.loc[testing_idx], target.loc[testing_idx])
        reference_score = self.scorer(reference, features.loc[testing_idx], target.loc[testing_idx])

        return model_score, reference_score

    def get_model(self, now, model, target: Target, features: Features):
        training_idx: Indices = list(range(now - self.training_window, now))
        testing_idx: Indices = list(range(now, now + self.testing_window + 1))

        model.fit(features.loc[training_idx], target.loc[training_idx])

        return model, features.loc[testing_idx], target.loc[testing_idx]
