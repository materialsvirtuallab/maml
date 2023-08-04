from __future__ import annotations

import os
import pickle
import unittest

from maml.sampling.clustering import BirchClustering

feature_file_path = os.path.join(
    os.path.dirname(__file__),
    "data",
    "M3GNet_features_MPF_2021_2_8_first10_features_test.pickle",
)
with open(feature_file_path, "rb") as f:
    MPF_2021_2_8_first10_features_test = pickle.load(f)
label_centers_file_path = os.path.join(
    os.path.dirname(__file__),
    "data",
    "MPF_2021_2_8_first10_Birch_label_centers.pickle",
)
with open(label_centers_file_path, "rb") as f:
    MPF_2021_2_8_first10_label_centers = pickle.load(f)


class BirchClusteringTest(unittest.TestCase):
    def setUp(self):
        self.Birch = BirchClustering(n=2, threshold_init=0.01)

    def test_fit(self):
        assert self.Birch == self.Birch.fit(
            MPF_2021_2_8_first10_features_test["M3GNet_features"]
        )

    def test_transform(self):
        clustering_results = self.Birch.transform(
            MPF_2021_2_8_first10_features_test["M3GNet_features"]
        )
        assert "labels" in clustering_results.keys()
        assert "label_centers" in clustering_results.keys()
        assert "PCAfeatures" in clustering_results.keys()
        assert (
            clustering_results["labels"] == np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 1])
        ).all()
        assert all(
            (
                (clustering_results["label_centers"].get(k) == v).all()
                for k, v in MPF_2021_2_8_first10_label_centers.items()
            )
        )
        assert (
            MPF_2021_2_8_first10_features_test["M3GNet_features"]
            == clustering_results["PCAfeatures"]
        ).all()


if __name__ == "__main__":
    unittest.main()