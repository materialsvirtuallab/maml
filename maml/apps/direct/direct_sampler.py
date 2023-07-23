from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from maml.describers.m3gnet import M3GNetStructure

from .clustering import BirchClustering
from .pca import PrincipleComponentAnalysis
from .stratified_sampling import SelectKFromClusters


class DIRECTSampler(Pipeline):
    """
    DImensionality REduction-Clustering-sTratified (DIRECT)
    sampling Pipeline.
    """

    def __init__(
        self,
        structure_encoder=M3GNetStructure(),
        scaler=StandardScaler(),
        pca=PrincipleComponentAnalysis(),
        weighting_PCs=True,
        clustering=BirchClustering(),
        select_k_from_clusters=SelectKFromClusters(),
    ):
        """
        Args:
        structure_encoder: Structure featurizer. It can be any encoder
            that takes in a list of N structures and returns a 2-D array
            of N*D features, where D is the fixed dimensionality of the
            feature vector. By default, the M3GNet formation energy model
            is used. Set this to False to skip the encoding step when needed.
        scaler: StandardScaler to perform normalization before PCA.
        pca: PCA for dimensionality reduction.
        weighting_PCs: Whether to weight PC with their explained variance.
        clustering: Clustering method to clustering based on PCs.
        select_k_from_clusters: Straitified sampling of k structures from
            each cluster.
        """
        self.structure_encoder = structure_encoder
        self.scaler = scaler
        pca.weighting_PCs = weighting_PCs
        self.pca = pca
        self.weighting_PCs = weighting_PCs
        self.clustering = clustering
        self.select_k_from_clusters = select_k_from_clusters
        steps = [
            (i.__class__.__name__, i)
            for i in [
                structure_encoder,
                scaler,
                pca,
                clustering,
                select_k_from_clusters,
            ]
            if i
        ]
        super().__init__(steps)
