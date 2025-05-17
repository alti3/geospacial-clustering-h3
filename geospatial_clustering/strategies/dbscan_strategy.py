from __future__ import annotations

from typing import List, Dict, Tuple
import numpy as np
import networkx as nx

from .base_strategy import ClusteringStrategy

try:
    from sklearn.cluster import DBSCAN  # type: ignore
    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover – optional dep
    SKLEARN_AVAILABLE = False


class DBSCANStrategy(ClusteringStrategy):
    """Density‑based clustering with a pre‑computed H3 distance matrix.

    Parameters
    ----------
    eps : int
        Maximum H3 grid distance to be considered *neighbours* (in H3 indexes), inclusive.
        Must match the ``distance_threshold`` used to build the graph for consistency.
    min_samples : int, default 1
        Minimum number of samples for a core point. Set to ``1`` so singleton clusters are
        preserved; raise as needed.
    """

    def __init__(self, eps: int, *, min_samples: int = 1):
        if not SKLEARN_AVAILABLE: # Check moved here
            raise RuntimeError("scikit‑learn must be installed for DBSCANStrategy.")
        self.eps = eps
        self.min_samples = min_samples

    def cluster(self, graph: nx.Graph, h3_index_to_coord: Dict[str, Tuple[float, float]]) -> List[List[str]]:
        # Map node IDs → row index in distance matrix
        nodes = list(graph.nodes)
        n = len(nodes)
        node_to_idx = {n: i for i, n in enumerate(nodes)}

        # Build a compressed sparse distance matrix (upper triangle) using numpy.
        # Because H3 distance is integer, we can store in uint8/uint16 as needed.
        max_possible = graph.graph["distance_threshold"]
        dtype = np.uint8 if max_possible < 256 else np.uint16
        dist_matrix = np.full((n, n), max_possible + 1, dtype=dtype)
        np.fill_diagonal(dist_matrix, 0)

        for u, v, d in graph.edges(data="weight"):
            i, j = node_to_idx[u], node_to_idx[v]
            dist_matrix[i, j] = dist_matrix[j, i] = d

        # scikit‑learn’s DBSCAN with metric="precomputed" expects float distances.
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="precomputed")
        labels = dbscan.fit_predict(dist_matrix.astype(float))

        clusters: Dict[int, List[str]] = {}
        for node_idx, label in enumerate(labels): # Iterate over labels directly
            node = nodes[node_idx] # Get node from original list
            # Original logic for handling noise points as singletons seems to create
            # a single cluster for ALL noise points.
            # If label == -1, it means it's a noise point.
            # To make them singleton clusters, each should get a unique cluster ID.
            # However, the original code put all noise points (-1) into a single list.
            # Let's stick to the original behavior for now, but this is a point of review.
            # If you want true singletons for noise, each noise point would be its own cluster.
            # The original code appends all noise points to clusters[-1].
            # The current code lumps noise points together. If you want them as individual clusters:
            # if label == -1:
            #     clusters[f"noise_{node}"] = [node] # Unique key for each noise point
            # else:
            #     clusters.setdefault(label, []).append(node)

            # Sticking to original logic for now:
            clusters.setdefault(label, []).append(node)
        return list(clusters.values())