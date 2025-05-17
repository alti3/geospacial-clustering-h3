from __future__ import annotations

from typing import List, Dict, Tuple
import numpy as np
import networkx as nx

from .base_strategy import ClusteringStrategy
from .dbscan_strategy import SKLEARN_AVAILABLE # For SKLEARN_AVAILABLE check

if SKLEARN_AVAILABLE:
    from sklearn.cluster import SpectralClustering # type: ignore

class SpectralClusteringStrategy(ClusteringStrategy):
    """Performs clustering using Spectral Clustering.

    This method uses the eigenvalues of a similarity matrix (affinity matrix)
    to perform dimensionality reduction before clustering in fewer dimensions.

    Uses scikit-learn's implementation.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form.
    assign_labels : {'kmeans', 'discretize', 'km'}, default='kmeans'
        The strategy to use to assign labels in the embedding space.
    affinity : str, default='precomputed'
        How to construct the affinity matrix. Since we provide a similarity matrix,
        this should be 'precomputed'.
        Note: SpectralClustering uses similarities, not distances.
    distance_threshold : int, optional
        The H3 distance threshold used for graph construction. Accepted for
        consistency in GeoClusterer, but not directly used by SpectralClustering
        parameters here as it operates on the precomputed similarity matrix.
    **spectral_kwargs :
        Other keyword arguments to pass to `sklearn.cluster.SpectralClustering`.
    """

    def __init__(self, *, n_clusters: int = 8, assign_labels: str = 'kmeans', affinity: str = 'precomputed', distance_threshold: int | None = None, **spectral_kwargs):
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn must be installed for SpectralClusteringStrategy.")
        self.n_clusters = n_clusters
        self.assign_labels = assign_labels
        self.affinity = affinity # Should be 'precomputed'
        # distance_threshold is accepted for API consistency.
        self.spectral_kwargs = spectral_kwargs

    def cluster(self, graph: nx.Graph, h3_index_to_coord: Dict[str, Tuple[float, float]]) -> List[List[str]]:
        nodes = list(graph.nodes)
        if not nodes:
            return []
        n_nodes = len(nodes)
        if n_nodes == 0:
            return []
        
        node_to_idx = {node_id: i for i, node_id in enumerate(nodes)}

        # Spectral Clustering uses a similarity matrix (affinity matrix).
        # Higher values mean more similar.
        # Using S = -D (negative distance) as in AffinityPropagationStrategy.
        # Or, S = max(D) - D or S = exp(-D^2 / (2 * sigma^2)). Let's try S = exp(-D / T).
        # For simplicity and consistency, let's try S = -D.
        # Diagonal elements S[i,i] should represent high self-similarity (e.g., 0 if S=-D, or 1 if S=exp(-D)).
        # Sklearn's SpectralClustering with 'precomputed' affinity expects a square matrix where S[i,j] is similarity.

        max_dist_val = graph.graph.get("distance_threshold", 1) # Default to 1 if not present
        
        # Initialize with a value indicating very low similarity for unconnected points
        # If S = -D, then for large D, S is large negative.
        default_unconnected_similarity = -(max_dist_val + 10) 

        similarity_matrix = np.full((n_nodes, n_nodes), default_unconnected_similarity, dtype=float)
        
        for u, v, d in graph.edges(data="weight"):
            i, j = node_to_idx[u], node_to_idx[v]
            # Convert distance to similarity. If d=0, similarity should be highest.
            # Using S = -D, so similarity_matrix[i,j] = -d
            similarity_matrix[i, j] = similarity_matrix[j, i] = -float(d if d is not None else max_dist_val + 1)
        
        # Self-similarity (diagonal)
        # For S = -D, self-similarity (distance 0) would be 0.
        # Some implementations might expect positive similarities.
        # If using Gaussian: np.exp(-dist_matrix ** 2 / (2. * sigma ** 2))
        # For S = -D, diagonal is 0. This should work.
        np.fill_diagonal(similarity_matrix, 0)

        # Ensure n_clusters is not more than n_samples for SpectralClustering
        effective_n_clusters = min(self.n_clusters, n_nodes)
        if effective_n_clusters <= 0: # if n_nodes is 0
             return []
        if n_nodes < 2 : # Not enough samples to form clusters or run spectral clustering meaningfully
            return [[node] for node in nodes] # Each node is its own cluster


        sc = SpectralClustering(
            n_clusters=effective_n_clusters,
            assign_labels=self.assign_labels,
            affinity=self.affinity, # Should be 'precomputed'
            random_state=self.spectral_kwargs.pop('random_state', None),
            **self.spectral_kwargs
        )
        
        try:
            labels = sc.fit_predict(similarity_matrix)
        except Exception as e:
            # If n_components is too large relative to n_samples, an error can occur.
            # Or other numerical issues with the similarity matrix.
            # Fallback or error logging
            print(f"Error during SpectralClustering: {e}. Matrix shape: {similarity_matrix.shape}, n_clusters: {effective_n_clusters}")
            # As a fallback, return singletons or connected components
            return [list(c) for c in nx.connected_components(graph)]


        clusters: Dict[int, List[str]] = {}
        for node_idx, label in enumerate(labels):
            # SpectralClustering does not produce -1 labels for noise.
            clusters.setdefault(label, []).append(nodes[node_idx])
        
        return list(clusters.values())
