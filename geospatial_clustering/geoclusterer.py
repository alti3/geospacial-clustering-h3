from __future__ import annotations

from typing import List, Tuple, Dict, Sequence, Union # Union added
import logging

import numpy as np
import h3
import networkx as nx

from .strategies import ClusteringStrategy, STRATEGY_REGISTRY # Relative import

logger = logging.getLogger(__name__)


class GeoClusterer:
    """Pipeline object that turns raw lat/lon coordinates into H3 clusters."""

    def __init__(
        self,
        coords: Sequence[Tuple[float, float]],
        resolution: int,
        *,
        distance_threshold: int = 1,
        strategy: Union[str, ClusteringStrategy] = "components", # Union for type hint
        **strategy_kwargs,
    ):
        self.coords = np.asarray(coords, dtype=float)
        if self.coords.ndim != 2 or self.coords.shape[1] != 2:
            raise ValueError("coords must be an (N, 2) array‑like of (lat, lon).")
        self.resolution = int(resolution)
        self.distance_threshold = int(distance_threshold)

        # Strategy instantiation
        if isinstance(strategy, str):
            strategy_key = strategy.lower()
            if strategy_key not in STRATEGY_REGISTRY:
                raise KeyError(
                    f"Unknown strategy '{strategy_key}'. Available: {list(STRATEGY_REGISTRY)}"
                )
            StrategyCls = STRATEGY_REGISTRY[strategy_key]
            # Pass distance_threshold to all strategies for consistency,
            # even if they don't all use it.
            # DBSCANStrategy specific args like min_samples are in strategy_kwargs
            self.strategy: ClusteringStrategy = StrategyCls(
                eps=distance_threshold, **strategy_kwargs # DBSCAN uses 'eps'
            ) if strategy_key == "dbscan" else StrategyCls(
                distance_threshold=distance_threshold, **strategy_kwargs
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
        self._h3_indexes: np.ndarray | None = None # Type hint was missing | None
        self._index_to_coord: Dict[str, Tuple[float, float]] | None = None

    def clusters(self) -> List[List[str]]:
        """Return clusters of H3 indexes using the configured strategy."""
        if self._graph is None:
            self._build_graph()
        assert self._graph is not None and self._index_to_coord is not None
        return self.strategy.cluster(self._graph, self._index_to_coord)

    def labelled_points(self):
        """Yield each original point with its cluster label (integer)."""
        if self._h3_indexes is None: # Ensure discretisation has happened
            self._discretise()
        assert self._h3_indexes is not None

        clusters_list = self.clusters()
        label_lookup: Dict[str, int] = {}
        for label, cluster_nodes in enumerate(clusters_list):
            for h3_idx_node in cluster_nodes:
                label_lookup[h3_idx_node] = label
        
        for i in range(len(self.coords)):
            lat, lon = self.coords[i]
            h3_idx_point = self._h3_indexes[i] # h3_idx for this specific point
            # A point's h3_idx might not be in label_lookup if it was a noise point
            # that didn't form a cluster or wasn't included in DBSCANStrategy's output structure
            # under its original H3 index if it was labeled -1.
            # The original DBSCAN puts all noise points into one cluster.
            # Let's ensure h3_idx_point is in label_lookup or handle default.
            point_label = label_lookup.get(h3_idx_point, -1) # Default to -1 if not found (e.g. noise)
            yield (lat, lon, point_label)


    def _discretise(self):
        """Convert coordinates to H3 indexes and populate lookup tables."""
        # h3.geo_to_h3 expects (lat, lon)
        # Correct usage of frompyfunc for vectorization
        # The lambda should take lat, lon and resolution
        h3_converter = np.frompyfunc(
            lambda lat, lon: h3.latlng_to_cell(lat, lon, self.resolution), 2, 1
        )
        h3_indexes_obj_array = h3_converter(self.coords[:, 0], self.coords[:, 1])
        
        # Ensure it's a flat array of H3 strings
        self._h3_indexes = np.asarray(h3_indexes_obj_array, dtype=object).flatten()

        # Map H3 index → first coordinate encountered (representative)
        # Ensure h3_indexes is a flat list of strings for zipping
        h3_indexes_list = self._h3_indexes.tolist()
        self._index_to_coord = {}
        for i, h3_idx_val in enumerate(h3_indexes_list):
            if h3_idx_val not in self._index_to_coord: # Store only the first coord for a given H3 cell
                 self._index_to_coord[h3_idx_val] = (self.coords[i, 0], self.coords[i, 1])


    def _build_graph(self):
        """Create a graph with nodes = unique H3 cells, edges within threshold."""
        if self._index_to_coord is None or self._h3_indexes is None: # Also check _h3_indexes
            self._discretise()
        assert self._h3_indexes is not None and self._index_to_coord is not None

        unique_indexes = np.unique(self._h3_indexes) # Use the full list of H3 indexes from points
        G = nx.Graph(distance_threshold=self.distance_threshold)
        G.add_nodes_from(unique_indexes)

        # Build edges by exploring k‑ring neighbourhoods to avoid O(N²) distance checks.
        processed_pairs = set() # To avoid duplicate checks and ensure h < n logic works correctly
        for h_node in unique_indexes:
            # h3.k_ring can return the center cell itself if k=0
            # We are interested in neighbours up to distance_threshold
            # h3.grid_disk is an alias for k_ring and might be clearer
            neighbours_and_self = h3.grid_disk(h_node, self.distance_threshold)
            for n_node in neighbours_and_self:
                if n_node == h_node: # Skip self-loops from k_ring perspective
                    continue
                
                # Ensure n_node is actually one of the unique H3 cells from our input data
                if n_node in G: # Check if potential neighbour is part of our unique cells
                    # To avoid duplicate edges and processing (u,v) and (v,u)
                    # we ensure one representation (e.g., smaller index first)
                    pair = tuple(sorted((h_node, n_node)))
                    if pair not in processed_pairs:
                        # h3.h3_distance can be expensive, do it only once per valid pair
                        dist = h3.grid_distance(h_node, n_node)
                        if dist is not None and dist <= self.distance_threshold:
                            G.add_edge(h_node, n_node, weight=dist)
                        processed_pairs.add(pair)
        self._graph = G