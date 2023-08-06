from __future__ import annotations

from maml.sampling.direct import DIRECTSampler


class TestDIRECTSampler:
    def setup(self):
        self.direct = DIRECTSampler(structure_encoder=None)

    def test_fit_transform(self, MPF_2021_2_8_first10_features_test):
        assert self.direct.fit_transform(MPF_2021_2_8_first10_features_test["M3GNet_features"])["selected_indexes"] == [
            9
        ]
