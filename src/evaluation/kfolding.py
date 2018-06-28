# coding=utf-8
import numpy as np

from data_primitives import SourceData
from evaluation.metrics import r2_scorer
from evaluation.model import ModelDefinition, ReferenceModel
from evaluation.scores import Scores, Results
from transformations import DatasetMaker


class KFoldEvaluator:
    def __init__(self, model_definition: ModelDefinition, dataset_maker: DatasetMaker, scorer=r2_scorer):
        self.scorer = scorer
        self.dataset_maker = dataset_maker
        self.model_definition = model_definition

    def results(self, df: SourceData) -> Results:
        data_folds = self.dataset_maker.k_fold_make(df)

        target = self.dataset_maker.transformer.transform_target(df)
        reference = ReferenceModel(target)

        model_scores = []
        reference_scores = []

        for training_data, testing_data in data_folds:
            model = self.model_definition.get_model()
            model.fit(training_data.features, training_data.target)

            model_score = self.scorer(model, testing_data.features, testing_data.target)
            reference_score = self.scorer(reference, testing_data.features, testing_data.target)

            model_scores.append(model_score)
            reference_scores.append(reference_score)

        model_results = Scores(np.array(model_scores))
        reference_results = Scores(np.array(reference_scores))

        return Results(reference_results, model_results)
