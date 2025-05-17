from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple, List
import networkx as nx # Added import

class ClusteringStrategy(ABC):
    """Abstract base class for clustering strategies."""

    @abstractmethod
    def cluster(self, graph: nx.Graph, h3_index_to_coord: Dict[str, Tuple[float, float]]) -> List[List[str]]:
        """Return a list of clusters, each cluster being a list of node IDs (H3 indexes)."""
        pass