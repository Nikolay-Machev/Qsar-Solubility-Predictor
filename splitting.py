"""Leakage-resistant grouped splitting utilities."""

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def grouped_train_test_indices(groups, test_size=0.2, seed=42):
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) < 2:
        raise ValueError("groups must be a one-dimensional sequence")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train, test = next(splitter.split(np.zeros(len(groups)), groups=groups))
    if set(groups[train]).intersection(groups[test]):
        raise RuntimeError("group leakage detected")
    return train, test
