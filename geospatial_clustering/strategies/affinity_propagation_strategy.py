from __future__ import annotations

from typing import List, Dict, Tuple
import numpy as np
import networkx as nx

from .base_strategy import ClusteringStrategy
from .dbscan_strategy import SKLEARN_AVAILABLE # For SKLEARN_AVAILABLE check

if SKLEARN_AVAILABLE:
    from sklearn.cluster import AffinityPropagation # type: ignore

class AffinityPropagationStrategy(ClusteringStrategy):
    """Performs clustering using Affinity Propagation.

    Affinity Propagation is a clustering algorithm based on the concept of
    "message passing" between data points. It does not require the number of
    clusters to be determined before running the algorithm.

    Uses scikit-learn's implementation.

    Parameters
    ----------
    damping : float, default=0.5
        Damping factor in the range `[0.5, 1.0)`.
    max_iter : int, default=200
        Maximum number of iterations.
    convergence_iter : int, default=15
        Number of iterations with no change in the number
        of estimated clusters that stops the convergence.
    preference : array-like of shape (n_samples,) or float, optional
        Preferences for each point - the larger the preference, the more likely
        it is to be chosen as an exemplar. If `None`, it will be set to the
        median of the input similarities.
    affinity : str, default='precomputed'
        Which affinity to use. Since we provide a similarity matrix (derived from
        distances), this should be 'precomputed'.
        Note: Affinity Propagation uses similarities, not distances. Higher values
        mean more similar. We will need to convert our distance matrix.
    **ap_kwargs :
        Other keyword arguments to pass to `sklearn.cluster.AffinityPropagation`.
    """

    def __init__(self, *, damping: float = 0.5, max_iter: int = 200, convergence_iter: int = 15, preference=None, affinity: str = 'precomputed', **ap_kwargs):
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn must be installed for AffinityPropagationStrategy.")
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.preference = preference
        self.affinity = affinity # Should be 'precomputed'
        self.ap_kwargs = ap_kwargs

    def cluster(self, graph: nx.Graph, h3_index_to_coord: Dict[str, Tuple[float, float]]) -> List[List[str]]:
        nodes = list(graph.nodes)
        if not nodes:
            return []
        n = len(nodes)
        node_to_idx = {node_id: i for i, node_id in enumerate(nodes)}

        # Affinity Propagation uses a similarity matrix (higher values = more similar).
        # Our graph edges store distances (lower values = more similar/closer).
        # We need to convert distances to similarities.
        # A common way: S = -D or S = exp(-D^2 / (2 * sigma^2)) or S = 1 / (1 + D)
        # S = -D is simplest if the algorithm handles negative similarities.
        # Sklearn's AffinityPropagation with 'precomputed' expects similarities where S[i,j] is similarity between i and j.
        # Let's use S = -D. Diagonal elements (self-similarity) are handled by `preference`.

        max_dist_val = graph.graph.get("distance_threshold", 0) # Reference
        # Initialize with a value indicating very low similarity for unconnected points
        # If S = -D, then for large D, S is large negative.
        default_unconnected_similarity = -(max_dist_val + 10) # Ensure it's less similar than actual connections

        similarity_matrix = np.full((n, n), default_unconnected_similarity, dtype=float)
        
        for u, v, d in graph.edges(data="weight"):
            i, j = node_to_idx[u], node_to_idx[v]
            similarity_matrix[i, j] = similarity_matrix[j, i] = -float(d) # Negative distance

        # Self-similarity (diagonal) is set by the `preference` parameter during AffinityPropagation fit.
        # If preference is not set, sklearn computes it from the median of similarities.
        # So, np.fill_diagonal(similarity_matrix, some_value) might conflict if preference is also None.
        # Let's allow sklearn to handle diagonal based on `preference`.
        # If Sii is not set and preference=None, AP calculates it.
        # If precomputed, S[i,i] are the preferences.
        # We should explicitly set the diagonal if `preference` is not being passed to AP, or ensure `preference` is an array.
        # For now, let AP handle it or user specify `preference` vector.

        # If no preference is given, it's often set to the median of the input similarities.
        # For S = -D, the diagonal S[i,i] should be 0 (or a chosen preference value).
        # Let's set diagonal to 0, implying distance to self is 0, so similarity is -0 = 0.
        np.fill_diagonal(similarity_matrix, 0)

        ap = AffinityPropagation(
            damping=self.damping,
            max_iter=self.max_iter,
            convergence_iter=self.convergence_iter,
            preference=self.preference,
            affinity=self.affinity, # Should be 'precomputed'
            random_state=self.ap_kwargs.pop('random_state', None), # Pass random_state if present
            **self.ap_kwargs
        )
        labels = ap.fit_predict(similarity_matrix)

        clusters: Dict[int, List[str]] = {}
        for node_idx, label in enumerate(labels):
            # AffinityPropagation does not produce -1 labels for noise by default.
            # Each point is assigned to a cluster.
            clusters.setdefault(label, []).append(nodes[node_idx])
        
        return list(clusters.values()) 