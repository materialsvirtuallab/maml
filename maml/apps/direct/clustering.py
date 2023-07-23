from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import Birch


class BirchClustering(BaseEstimator, TransformerMixin):
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
        return self

    def transform(self, PCAfeatures):
        model = Birch(
            n_clusters=self.n, threshold=self.threshold_init, **self.kwargs
        ).fit(PCAfeatures)
        while (
            len(model.subcluster_labels_) < self.n
        ):  # decrease threshold until desired n clusters is achieved
            print(
                f"Birch threshold of {self.threshold_init} gives {len(model.subcluster_labels_)} clusters."
            )
            self.threshold_init = (
                self.threshold_init / self.n * len(model.subcluster_labels_)
            )
            model = Birch(
                n_clusters=self.n, threshold=self.threshold_init, **self.kwargs
            ).fit(PCAfeatures)
        labels = model.predict(PCAfeatures)
        self.model = model
        print(
            f"Birch threshold of {self.threshold_init} gives {len(model.subcluster_labels_)} clusters."
        )
        label_centers = {
            l: c for l, c in zip(model.subcluster_labels_, model.subcluster_centers_)
        }
        return {
            "labels": labels,
            "label_centers": label_centers,
            "PCAfeatures": PCAfeatures,
        }
