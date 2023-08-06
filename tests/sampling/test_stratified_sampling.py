from __future__ import annotations

from maml.sampling.stratified_sampling import SelectKFromClusters


class TestSelectKFromClusters:
    def setup(self):
        self.selector_uni = SelectKFromClusters(k=2, allow_duplicate=False)
        self.selector_dup = SelectKFromClusters(k=2, allow_duplicate=True)

    def test_fit(self, Birch_results):
        assert self.selector_uni == self.selector_uni.fit(Birch_results)
        assert self.selector_dup == self.selector_dup.fit(Birch_results)

    def test_transform(self, Birch_results):
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
