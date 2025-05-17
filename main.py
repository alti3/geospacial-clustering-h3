from __future__ import annotations
"""Geospatial Clustering with Uber H3
------------------------------------
This module provides a flexible, high‑performance pipeline for clustering point data on a
spherical surface by discretising into H3 hexagonal cells, constructing a graph based on
H3 grid distance, and applying a pluggable clustering algorithm (strategy pattern).

Dependencies
------------
* h3‑py >= 4.0 (``pip install h3``)
* numpy >= 1.23 (``pip install numpy``)
* networkx >= 3.0 (``pip install networkx``)
* scikit‑learn >= 1.4 (optional, for DBSCAN strategy – ``pip install scikit­‑learn``)

Core Concepts
-------------
1. **Discretisation** – Continuous latitude/longitude pairs are converted to H3 cell indexes
   at a user‑defined resolution.
2. **Graph Construction** – An undirected graph is built where each node is a unique H3
   cell and an edge is added when the *grid distance* (``h3_distance``) between two cells
   is ≤ ``distance_threshold``.
3. **Strategy Pattern** – Clustering is delegated to a ``ClusteringStrategy`` object which
   operates on the graph (or its adjacency matrix). Two strategies are included:
     * ``ConnectedComponentsStrategy`` – fast, no external deps.
     * ``DBSCANStrategy`` – density‑based clustering using scikit‑learn’s ``DBSCAN`` with a
       pre‑computed distance matrix.

The design allows you to drop‑in alternative strategies (e.g. Louvain, Leiden, HDBSCAN)."""
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Sequence
import logging

import numpy as np
import h3
import networkx as nx

try:
    from sklearn.cluster import DBSCAN  # type: ignore

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover – optional dep
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strategy Pattern for clustering
# ---------------------------------------------------------------------------


class ClusteringStrategy(ABC):
    """Abstract base class for clustering strategies."""

    @abstractmethod
    def cluster(self, graph: nx.Graph, h3_index_to_coord: Dict[str, Tuple[float, float]]):
        """Return a list of clusters, each cluster being a list of node IDs (H3 indexes)."""


class ConnectedComponentsStrategy(ClusteringStrategy):
    """Clusters are weakly‑connected components in the graph."""

    def cluster(self, graph: nx.Graph, h3_index_to_coord):  # noqa: D401 – plain function
        return [list(c) for c in nx.connected_components(graph)]


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
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit‑learn must be installed for DBSCANStrategy.")
        self.eps = eps
        self.min_samples = min_samples

    def cluster(self, graph: nx.Graph, h3_index_to_coord):
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
        for node, label in zip(nodes, labels):
            if label == -1:
                # Noise points form singleton clusters labelled by their own index
                clusters.setdefault(label, []).append(node)
            else:
                clusters.setdefault(label, []).append(node)
        return list(clusters.values())


# Registry for convenience ---------------------------------------------------

STRATEGY_REGISTRY = {
    "components": ConnectedComponentsStrategy,
    "dbscan": DBSCANStrategy,
}


# ---------------------------------------------------------------------------
# GeoClusterer: orchestrates work from raw coordinates → clusters
# ---------------------------------------------------------------------------


class GeoClusterer:
    """Pipeline object that turns raw lat/lon coordinates into H3 clusters."""

    def __init__(
        self,
        coords: Sequence[Tuple[float, float]],
        resolution: int,
        *,
        distance_threshold: int = 1,
        strategy: str | ClusteringStrategy = "components",
        **strategy_kwargs,
    ):
        self.coords = np.asarray(coords, dtype=float)
        if self.coords.ndim != 2 or self.coords.shape[1] != 2:
            raise ValueError("coords must be an (N, 2) array‑like of (lat, lon).")
        self.resolution = int(resolution)
        self.distance_threshold = int(distance_threshold)

        # Strategy instantiation ------------------------------------------------
        if isinstance(strategy, str):
            if strategy.lower() not in STRATEGY_REGISTRY:
                raise KeyError(
                    f"Unknown strategy '{strategy}'. Available: {list(STRATEGY_REGISTRY)}"
                )
            StrategyCls = STRATEGY_REGISTRY[strategy.lower()]
            self.strategy: ClusteringStrategy = StrategyCls(
                distance_threshold, **strategy_kwargs
            )
        elif isinstance(strategy, ClusteringStrategy):
            self.strategy = strategy
        else:
            raise TypeError("strategy must be a string key or ClusteringStrategy instance.")

        logger.info(
            "Initialised GeoClusterer with %d points, resolution %d, threshold %d, strategy %s",
            len(self.coords),
            self.resolution,
            self.distance_threshold,
            self.strategy.__class__.__name__,
        )

        self._graph: nx.Graph | None = None
        self._h3_indexes: np.ndarray | None = None
        self._index_to_coord: Dict[str, Tuple[float, float]] | None = None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def clusters(self) -> List[List[str]]:
        """Return clusters of H3 indexes using the configured strategy."""
        if self._graph is None:
            self._build_graph()
        assert self._graph is not None and self._index_to_coord is not None
        return self.strategy.cluster(self._graph, self._index_to_coord)

    def labelled_points(self):
        """Yield each original point with its cluster label (integer)."""
        clusters = self.clusters()
        label_lookup = {
            h3_idx: label for label, cluster in enumerate(clusters) for h3_idx in cluster
        }
        for (lat, lon), h3_idx in zip(self.coords, self._h3_indexes):
            yield (lat, lon, label_lookup[h3_idx])

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _discretise(self):
        """Convert coordinates to H3 indexes and populate lookup tables."""
        h3_indexes = np.frompyfunc(
            lambda lat, lon: h3.geo_to_h3(lat, lon, self.resolution), 2, 1
        )(self.coords[:, 0], self.coords[:, 1]).astype(object)
        self._h3_indexes = h3_indexes
        # Map H3 index → first coordinate encountered (representative)
        self._index_to_coord = {
            h3_idx: (lat, lon)
            for (lat, lon), h3_idx in zip(self.coords.tolist(), h3_indexes.tolist())
        }

    def _build_graph(self):
        """Create a graph with nodes = unique H3 cells, edges within threshold."""
        if self._index_to_coord is None:
            self._discretise()
        assert self._h3_indexes is not None and self._index_to_coord is not None

        unique_indexes = np.unique(self._h3_indexes)
        G = nx.Graph(distance_threshold=self.distance_threshold)
        G.add_nodes_from(unique_indexes)

        # Build edges by exploring k‑ring neighbourhoods to avoid O(N²) distance checks.
        for h in unique_indexes:
            neighbours = h3.k_ring(h, self.distance_threshold)
            for n in neighbours:
                if n in G and h < n:  # undirected; ensure single edge
                    d = h3.h3_distance(h, n)
                    if d is not None and d <= self.distance_threshold:
                        G.add_edge(h, n, weight=d)
        self._graph = G


# ---------------------------------------------------------------------------
# Example usage (as CLI)
# ---------------------------------------------------------------------------

def _example():  # pragma: no cover
    import random

    # Generate synthetic points around two centres
    random.seed(123)
    pts1 = [
        (37.775 + random.uniform(-0.01, 0.01), -122.42 + random.uniform(-0.01, 0.01))
        for _ in range(50)
    ]
    pts2 = [
        (34.052 + random.uniform(-0.01, 0.01), -118.243 + random.uniform(-0.01, 0.01))
        for _ in range(50)
    ]
    points = pts1 + pts2

    clusterer = GeoClusterer(
        points, resolution=9, distance_threshold=1, strategy="components"
    )
    clusters = clusterer.clusters()
    print(f"Found {len(clusters)} clusters")
    for i, clus in enumerate(clusters):
        print(f"Cluster {i}: size={len(clus)}")


if __name__ == "__main__":  # pragma: no cover
    _example()
