"""Fast regression tests for leakage-resistant grouped splitting."""

import unittest
import numpy as np

from splitting import grouped_train_test_indices


class GroupedSplitTests(unittest.TestCase):
    def test_groups_never_cross_partitions(self):
        groups = np.array(["a", "a", "b", "c", "c", "d", "e", "f"])
        train, test = grouped_train_test_indices(groups, test_size=0.25, seed=7)
        self.assertFalse(set(groups[train]).intersection(groups[test]))
        self.assertEqual(sorted(np.concatenate([train, test])), list(range(len(groups))))

    def test_split_is_reproducible(self):
        groups = [str(index // 2) for index in range(20)]
        first = grouped_train_test_indices(groups, seed=42)
        second = grouped_train_test_indices(groups, seed=42)
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(np.array_equal(first[1], second[1]))


if __name__ == "__main__":
    unittest.main()
