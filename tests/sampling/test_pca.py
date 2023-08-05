from __future__ import annotations

import os
import pickle
import unittest

from maml.sampling.pca import PrincipalComponentAnalysis

feature_file_path = os.path.join(
    os.path.dirname(__file__),
    "data",
    "M3GNet_features_MPF_2021_2_8_first10_features_test.pickle",
)
with open(feature_file_path, "rb") as f:
    MPF_2021_2_8_first10_features_test = pickle.load(f)

PC_weighted_file_path = os.path.join(
    os.path.dirname(__file__),
    "data",
    "MPF_2021_2_8_first10_test_PC_weighted.pickle",
)
with open(PC_weighted_file_path, "rb") as f:
    PC_weighted = pickle.load(f)

PC_unweighted_file_path = os.path.join(
    os.path.dirname(__file__),
    "data",
    "MPF_2021_2_8_first10_test_PC_unweighted.pickle",
)
with open(PC_unweighted_file_path, "rb") as f:
    PC_unweighted = pickle.load(f)


class PrincipalComponentAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.pca_weightedPC = PrincipalComponentAnalysis(weighting_PCs=True)
        self.pca_unweightedPC = PrincipalComponentAnalysis(weighting_PCs=False)

    def test_fit(self):
        assert self.pca_weightedPC == self.pca_weightedPC.fit(
            MPF_2021_2_8_first10_features_test["M3GNet_features_normalized"]
        )
        assert self.pca_unweightedPC == self.pca_unweightedPC.fit(
            MPF_2021_2_8_first10_features_test["M3GNet_features_normalized"]
        )

    def test_transform(self):
        assert (
            self.pca_weightedPC.fit_transform(
                MPF_2021_2_8_first10_features_test["M3GNet_features_normalized"]
            )
            == PC_weighted
        ).all
        assert (
            self.pca_unweightedPC.fit_transform(
                MPF_2021_2_8_first10_features_test["M3GNet_features_normalized"]
            )
            == PC_unweighted
        ).all


if __name__ == "__main__":
    unittest.main()
