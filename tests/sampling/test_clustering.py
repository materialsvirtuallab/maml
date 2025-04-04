from __future__ import annotations

import numpy as np

from maml.sampling.clustering import BirchClustering


class TestBirchClustering:
    def setup_method(self):
        self.Birch = BirchClustering(n=2, threshold_init=0.01)
        self.Birch_t002 = BirchClustering(n=10, threshold_init=0.1)

    def test_fit(self, PC_weighted):
        assert self.Birch == self.Birch.fit(PC_weighted)

    def test_transform(self, Birch_results, PC_weighted):
        clustering_results = self.Birch.transform(PC_weighted)
        assert "labels" in clustering_results.keys()
        assert "label_centers" in clustering_results.keys()
        assert "PCAfeatures" in clustering_results.keys()
        assert (clustering_results["labels"] == Birch_results["labels"]).all()
        assert all(
            ((clustering_results["label_centers"].get(k) == v).all() for k, v in Birch_results["label_centers"].items())
        )
        assert (clustering_results["PCAfeatures"] == Birch_results["PCAfeatures"]).all()
        clustering_results_t002 = self.Birch_t002.transform(PC_weighted)
        assert len(set(clustering_results_t002["labels"])) == 10
