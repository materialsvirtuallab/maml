from __future__ import annotations

from maml.sampling.stratified_sampling import SelectKFromClusters

import pytest


class TestSelectKFromClusters:
    def setup_method(self):
        self.selector_uni = SelectKFromClusters(k=2, allow_duplicate=False)
        self.selector_dup = SelectKFromClusters(k=2, allow_duplicate=True)
        self.selector_rand = SelectKFromClusters(k=2, allow_duplicate=False, selection_criteria="random")
        self.selector_small = SelectKFromClusters(
            k=2,
            allow_duplicate=False,
            selection_criteria="smallest",
            n_sites=range(10),
        )
        self.selector_small_wrong_n_sites = SelectKFromClusters(
            k=2,
            allow_duplicate=False,
            selection_criteria="smallest",
            n_sites=range(11),
        )

    def test_exceptions(self, Birch_results):
        with pytest.raises(ValueError, match="Invalid selection_criteria"):
            SelectKFromClusters(selection_criteria="whatever")
        with pytest.raises(ValueError, match="n_sites must be provided"):
            SelectKFromClusters(selection_criteria="smallest")
        with pytest.raises(ValueError, match="n_sites must have same length as features"):
            self.selector_small_wrong_n_sites.transform(Birch_results)

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
        assert len(self.selector_rand.transform(Birch_results)["selected_indexes"]) == 3
        assert self.selector_small.transform(Birch_results)["selected_indexes"] == [
            0,
            1,
            7,
        ]
