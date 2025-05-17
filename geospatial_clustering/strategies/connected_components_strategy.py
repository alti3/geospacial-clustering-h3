from __future__ import annotations

from typing import List, Dict, Tuple
import networkx as nx

from .base_strategy import ClusteringStrategy

class ConnectedComponentsStrategy(ClusteringStrategy):
    """Clusters are weakly‑connected components in the graph."""

    def __init__(self, distance_threshold: int, **kwargs): # Added to match GeoClusterer signature
        """
        Initialize ConnectedComponentsStrategy.
        The distance_threshold and kwargs are accepted for consistency with other strategies
        but are not used by this specific strategy.
        """
        pass


    def cluster(self, graph: nx.Graph, h3_index_to_coord: Dict[str, Tuple[float, float]]) -> List[List[str]]:  # noqa: D401 – plain function
        return [list(c) for c in nx.connected_components(graph)]