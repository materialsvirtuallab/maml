"""Clustering methods."""
from __future__ import annotations

import logging

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import Birch

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BirchClustering(BaseEstimator, TransformerMixin):
    """ "Birch Clustering as one step of the DIRECT pipeline."""

    def __init__(self, n: int = 1, threshold_init=0.5, **kwargs):
        """
        Args:
            n: Clustering the PCs into n clusters.
            threshold_init: The initial radius of the subcluster obtained by merging
                a new sample and the closest subcluster should be lesser than
                the threshold. Otherwise a new subcluster is started. See details in:
                https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html.
                Users may tune this value for desired performance of birch, while 0.5
                is generally a good starting point, and some automatic tuning is done
                with our built-in codes to achieve n clusters.
        """
        self.n = n
        self.threshold_init = threshold_init
        self.kwargs = kwargs

    def fit(self, X, y=None):
        """
        Place holder for fit API.

        Args:
            X: Any inputs
            y: Any outputs

        Returns: self
        """
        return self

    def transform(self, PCAfeatures):
        """
        Perform Birch Clustering to an array of input PCA features.

        Args:
            PCAfeatures: An array of PCA features.

        Returns:
            A dict of Birch Clustering results, including labels of each
            PCA feature, centroid positions of each cluster in PCA feature s
            pace, and the array of input PCA features.
        """
        model = Birch(n_clusters=self.n, threshold=self.threshold_init, **self.kwargs).fit(PCAfeatures)
        while len(model.subcluster_labels_) < self.n:  # decrease threshold until desired n clusters is achieved
            logger.info(f"Birch threshold of {self.threshold_init} gives {len(model.subcluster_labels_)} clusters.")
            self.threshold_init = self.threshold_init / self.n * len(model.subcluster_labels_)
            model = Birch(n_clusters=self.n, threshold=self.threshold_init, **self.kwargs).fit(PCAfeatures)
        labels = model.predict(PCAfeatures)
        self.model = model
        logger.info(f"Birch threshold of {self.threshold_init} gives {len(model.subcluster_labels_)} clusters.")
        label_centers = dict(zip(model.subcluster_labels_, model.subcluster_centers_))
        return {
            "labels": labels,
            "label_centers": label_centers,
            "PCAfeatures": PCAfeatures,
        }
