from __future__ import annotations

from maml.sampling.direct import DIRECTSampler
from maml.sampling.clustering import BirchClustering


class TestDIRECTSampler:
    def setup_method(self):
        self.direct_fixed_n = DIRECTSampler(structure_encoder=None, clustering=BirchClustering(n=1))
        self.direct_fixed_t = DIRECTSampler(
            structure_encoder=None,
            clustering=BirchClustering(n=None, threshold_init=0.5),
        )

    def test_fit_transform(self, MPF_2021_2_8_first10_features_test):
        result_fixed_n = self.direct_fixed_n.fit_transform(MPF_2021_2_8_first10_features_test["M3GNet_features"])
        result_fixed_t = self.direct_fixed_t.fit_transform(MPF_2021_2_8_first10_features_test["M3GNet_features"])
        assert result_fixed_n["selected_indexes"] == [9]
        assert result_fixed_t["selected_indexes"] == [0, 6, 7, 8, 9]
