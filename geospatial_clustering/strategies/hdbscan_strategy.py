from __future__ import annotations

from typing import List, Dict, Tuple
import numpy as np
import networkx as nx

from .base_strategy import ClusteringStrategy

try:
    import hdbscan # type: ignore
    HDBSCAN_AVAILABLE = True
except ImportError: # pragma: no cover
    HDBSCAN_AVAILABLE = False


class HDBSCANStrategy(ClusteringStrategy):
    """Hierarchical Density-Based Spatial Clustering of Applications with Noise.

    Uses the hdbscan library. HDBSCAN can find clusters of varying densities
    and is generally more robust to parameter selection than DBSCAN.

    Parameters
    ----------
    min_cluster_size : int, default 5
        The minimum number of samples in a group for that group to be
        considered a cluster.
    min_samples : int, optional
        The number of samples in a neighbourhood for a point to be considered
        as a core point. This includes the point itself. If `None`, defaults
        to `min_cluster_size`.
    cluster_selection_epsilon : float, default 0.0
        A distance threshold. Clusters below this value will be merged.
        Used for HDBSCAN's cluster extraction.
    metric : str, default 'precomputed'
        The metric to use when calculating distance between instances in a
        feature array. Since we provide a distance matrix, this should be
        'precomputed'.
    distance_threshold : int, optional
        The H3 distance threshold used for graph construction. Accepted for
        consistency in GeoClusterer, but not directly used by HDBSCAN
        parameters here as it operates on the precomputed distance matrix.
    **hdbscan_kwargs :
        Other keyword arguments to pass to the `hdbscan.HDBSCAN` constructor.
    """

    def __init__(self, *, min_cluster_size: int = 5, min_samples: int | None = None, cluster_selection_epsilon: float = 0.0, metric: str = 'precomputed', distance_threshold: int | None = None, **hdbscan_kwargs):
        if not HDBSCAN_AVAILABLE:
            raise RuntimeError("hdbscan library must be installed for HDBSCANStrategy.")
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric
        # distance_threshold is accepted for API consistency.
        self.hdbscan_kwargs = hdbscan_kwargs

    def cluster(self, graph: nx.Graph, h3_index_to_coord: Dict[str, Tuple[float, float]]) -> List[List[str]]:
        nodes = list(graph.nodes)
        if not nodes:
            return []
        n = len(nodes)
        node_to_idx = {node_id: i for i, node_id in enumerate(nodes)}

        # HDBSCAN expects a distance matrix.
        # Use the graph's distance_threshold as the max distance for unconnected components.
        # Or, more robustly, use a large enough number if not specified or if components can be further.
        # For HDBSCAN, 'infinity' is often represented by a large number if metric is precomputed.
        # However, edges in our graph only exist if <= distance_threshold.
        # Points not connected by an edge up to distance_threshold are effectively "infinitely" far apart
        # for the purpose of the graph-based clustering.
        # The distance_threshold from the graph is a good candidate for `max_dist` in DBSCAN,
        # but HDBSCAN handles this differently with its hierarchy.
        # We will build the full distance matrix.
        
        max_dist_val = graph.graph.get("distance_threshold", 0)
        # Create a dense matrix initialized with a value indicating "no direct path"
        # or a distance larger than any expected cluster diameter if not using cluster_selection_epsilon.
        # A value slightly larger than max_dist_val from the graph is reasonable.
        # If cluster_selection_epsilon is used, it acts on the condensed tree, not this matrix directly.
        
        # Initialize with a value that signifies "far" for HDBSCAN if no edge
        # This could be a large number, or related to max_possible_h3_distance if known
        # For precomputed, HDBSCAN works with the provided distances.
        # Let's use a value slightly larger than any actual edge weight.
        # If distance_threshold is small, e.g. 1 or 2, use a slightly larger default for unconnected points
        # or rely on hdbscan internal handling of distance matrices.
        
        # Using graph.graph["distance_threshold"] as a reference for "far" if no edge
        # This should be fine as HDBSCAN builds its hierarchy from these distances.
        # If two nodes are not connected in the graph (up to distance_threshold), their
        # distance in this matrix will be > distance_threshold used for graph construction.

        # Let's ensure unconnected pairs have a distance that won't make them part of small epsilon clusters
        # if not using cluster_selection_epsilon. A common practice is to set them to a value
        # larger than any valid clustering distance you're interested in.
        # For HDBSCAN, the actual values matter. If no edge, they are "far".
        # The largest possible H3 distance might be too large and could skew things.
        # Max value from existing edges + 1 or a conventional large number.
        # Max H3 distance for a given resolution is not easily determined without full H3 specs.
        # We will use distance_threshold + 1 from graph config.
        default_unconnected_dist = max_dist_val + 1

        dist_matrix = np.full((n, n), default_unconnected_dist, dtype=float)
        np.fill_diagonal(dist_matrix, 0)

        for u, v, d in graph.edges(data="weight"):
            i, j = node_to_idx[u], node_to_idx[v]
            dist_matrix[i, j] = dist_matrix[j, i] = float(d)


        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric=self.metric,
            **self.hdbscan_kwargs
        )
        labels = clusterer.fit_predict(dist_matrix)

        clusters: Dict[int, List[str]] = {}
        for node_idx, label in enumerate(labels):
            if label == -1:  # Noise points
                # HDBSCAN conventionally labels noise as -1.
                # We can choose to put them in a single "noise" cluster or ignore them.
                # To match DBSCAN's current behavior of grouping noise:
                clusters.setdefault(label, []).append(nodes[node_idx])
                # If you want to exclude noise:
                # continue
            else:
                clusters.setdefault(label, []).append(nodes[node_idx])
        
        return list(clusters.values()) 