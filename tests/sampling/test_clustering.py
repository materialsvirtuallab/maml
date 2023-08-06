from __future__ import annotations

import numpy as np
import os
import pickle
import unittest

from maml.sampling.clustering import BirchClustering

PC_weighted_file_path = os.path.join(
    os.path.dirname(__file__),
    "data",
    "MPF_2021_2_8_first10_test_PC_weighted.pickle",
)
with open(PC_weighted_file_path, "rb") as f:
    PC_weighted = pickle.load(f)

Birch_result_file_path = os.path.join(
    os.path.dirname(__file__),
    "data",
    "MPF_2021_2_8_first10_Birch_results.pickle",
)
with open(Birch_result_file_path, "rb") as f:
    Birch_results = pickle.load(f)


class TestBirchClustering(unittest.TestCase):
    def setUp(self):
        self.Birch = BirchClustering(n=2, threshold_init=0.01)

    def test_fit(self):
        assert self.Birch == self.Birch.fit(PC_weighted)

    def test_transform(self):
        clustering_results = self.Birch.transform(PC_weighted)
        assert "labels" in clustering_results.keys()
        assert "label_centers" in clustering_results.keys()
        assert "PCAfeatures" in clustering_results.keys()
        assert (clustering_results["labels"] == Birch_results["labels"]).all()
        assert all(
            (
                (clustering_results["label_centers"].get(k) == v).all()
                for k, v in Birch_results["label_centers"].items()
            )
        )
        assert (clustering_results["PCAfeatures"] == Birch_results["PCAfeatures"]).all()


if __name__ == "__main__":
    unittest.main()
