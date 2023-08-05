from __future__ import annotations

import os
import pickle
import unittest

from maml.sampling.stratified_sampling import SelectKFromClusters

Birch_result_file_path = os.path.join(
    os.path.dirname(__file__),
    "data",
    "MPF_2021_2_8_first10_Birch_results.pickle",
)
with open(Birch_result_file_path, "rb") as f:
    Birch_results = pickle.load(f)


class SelectKFromClustersTest(unittest.TestCase):
    def setUp(self):
        self.selector_uni = SelectKFromClusters(k=2, allow_duplicate=False)
        self.selector_dup = SelectKFromClusters(k=2, allow_duplicate=True)

    def test_fit(self):
        assert self.selector_uni == self.selector_uni.fit(Birch_results)
        assert self.selector_dup == self.selector_dup.fit(Birch_results)

    def test_transform(self):
        assert self.selector_uni.transform(Birch_results)["selected_indexes"] == [
            9,
            6,
            7,
        ]
        assert self.selector_dup.transform(Birch_results)["selected_indexes"] == [
            9,
            6,
            7,
            7,
        ]


if __name__ == "__main__":
    unittest.main()
