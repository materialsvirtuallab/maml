from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA


class PrincipleComponentAnalysis(BaseEstimator, TransformerMixin):
    def __init__(self, weighting_PCs=True):
        """
        Args:
            weighting_PCs: Whether to weight PCs with explained variances.
        """
        self.pca = PCA()
        self.weighting_PCs = weighting_PCs

    def fit(self, normalized_features):
        self.pca.fit(normalized_features)
        return self

    def transform(self, normalized_features):
        m = len([e for e in self.pca.explained_variance_ if e > 1])
        explained_variance = self.pca.explained_variance_ratio_
        print(
            f"Selected first {m} PCs, explaining {100 * sum(explained_variance[:m]):.2f}% variance"
        )
        if not self.weighting_PCs:
            return self.pca.transform(normalized_features)[:, :m]
        return self.pca.transform(normalized_features)[:, :m] * explained_variance[:m]
