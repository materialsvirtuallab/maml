from __future__ import annotations

import os
import pickle
import unittest

from maml.sampling.direct import DIRECTSampler

feature_file_path = os.path.join(
    os.path.dirname(__file__),
    "data",
    "M3GNet_features_MPF_2021_2_8_first10_features_test.pickle",
)
with open(feature_file_path, "rb") as f:
    MPF_2021_2_8_first10_features_test = pickle.load(f)


class DIRECTSamplerTest(unittest.TestCase):
    def setUp(self):
        self.direct = DIRECTSampler(structure_encoder=None)

    def test_fit_transform(self):
        assert self.direct.fit_transform(MPF_2021_2_8_first10_features_test['M3GNet_features'])['selected_indexes'] == [
            9]


if __name__ == "__main__":
    unittest.main()