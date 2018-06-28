# coding=utf-8
from typing import List

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from data_primitives import SourceData, Indices, Split


class Splitter:
    def __init__(self, seed: int = None, ratio=0.2, k: int = 5):
        self.k = k
        self.ratio = ratio
        self.seed = seed

    def train_test_indices(self, df: SourceData) -> Split:
        idx = df.index.values
        train_idx, test_idx = train_test_split(idx, test_size=self.ratio, random_state=self.seed)
        return Split(Indices(train_idx), Indices(test_idx))

    def k_fold_indices(self, df: SourceData) -> List[Split]:
        kfold = KFold(self.k, shuffle=True, random_state=self.seed)
        idx = df.index.values
        splits = [Split(Indices(idx[train_idx]), Indices(idx[test_idx])) for train_idx, test_idx in kfold.split(idx)]
        return splits
