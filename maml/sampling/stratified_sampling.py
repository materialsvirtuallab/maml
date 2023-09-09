"""Implementation of stratefied sampling approaches."""
from __future__ import annotations

import logging
import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SelectKFromClusters(BaseEstimator, TransformerMixin):
    """Wrapper around selection of k data from each cluster."""

    def __init__(self, k: int = 1, allow_duplicate=False):
        """
        Args:
            k: Select k structures from each cluster.
            allow_duplicate: Whether structures are allowed to be
                selected over once.
        """
        self.k = k
        self.allow_duplicate = allow_duplicate

    def fit(self, X, y=None):
        """
        Fit the model.

        Args:
            X: Input features
            y: Target.
        """
        return self

    def transform(self, clustering_data: dict):
        """
        Perform stratified sampling of data from each cluster
        based on clustering results.

        Args:
            clustering_data: Results from clustering in a dict. The dict
                should at least contain "PCAfeatures" and their respective
                "labels" of belonged clusters. The positions of centroid
                for each cluster should also be provided with "label_centers",
                with which data in each cluster can be ranked according to
                their Euclidean distance to centroid and then selected by
                interval for optimal coverage.

        Returns:
            A dict with "PCAfeatures" used in clustering and "selected_indexes"
                as the indexes of DIRECT sampled structures.
        """
        if any(key not in clustering_data for key in ["labels", "PCAfeatures"]):
            raise Exception(
                "The data returned by clustering step should at least provide label and feature information."
            )
        if "label_centers" not in clustering_data:
            warnings.warn(
                "Centroid location is not provided, so random selection from each cluster will be performed, "
                "which will likely still significantly outperform manual sampling in terms of feature coverage. "
            )

        selected_indexes = []
        for label in set(clustering_data["labels"]):
            indexes_same_label = np.where(label == clustering_data["labels"])[0]
            features_same_label = clustering_data["PCAfeatures"][indexes_same_label]
            n_same_label = len(features_same_label)
            if "label_centers" in clustering_data:
                center_same_label = clustering_data["label_centers"][label]
                distance_to_center = np.linalg.norm(features_same_label - center_same_label, axis=1).reshape(
                    len(indexes_same_label)
                )
                select_k_indexes = [int(i) for i in np.linspace(0, n_same_label - 1, self.k)]
                selected_indexes.extend(
                    indexes_same_label[np.argpartition(distance_to_center, select_k_indexes)[select_k_indexes]]
                )
            else:
                selected_indexes.extend(indexes_same_label[np.random.randint(n_same_label, size=self.k)])
        n_duplicate = len(selected_indexes) - len(set(selected_indexes))
        if not self.allow_duplicate and n_duplicate > 0:
            selected_indexes = list(set(selected_indexes))
        elif self.allow_duplicate and n_duplicate > 0:
            warnings.warn(f"There are {n_duplicate} duplicated selections.")
        logger.info(f"Finally selected {len(selected_indexes)} configurations.")
        return {
            "PCAfeatures": clustering_data["PCAfeatures"],
            "selected_indexes": selected_indexes,
        }
