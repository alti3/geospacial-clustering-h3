from __future__ import annotations

from .base_strategy import ClusteringStrategy
from .connected_components_strategy import ConnectedComponentsStrategy
from .dbscan_strategy import DBSCANStrategy, SKLEARN_AVAILABLE # Expose SKLEARN_AVAILABLE if needed elsewhere
from .affinity_propagation_strategy import AffinityPropagationStrategy
from .hdbscan_strategy import HDBSCANStrategy, HDBSCAN_AVAILABLE
from .spectral_clustering_strategy import SpectralClusteringStrategy

# Registry for convenience
STRATEGY_REGISTRY = {
    "components": ConnectedComponentsStrategy,
    "dbscan": DBSCANStrategy,
    "affinity": AffinityPropagationStrategy,
    "hdbscan": HDBSCANStrategy,
    "spectral": SpectralClusteringStrategy,
}

__all__ = [
    "ClusteringStrategy",
    "ConnectedComponentsStrategy",
    "DBSCANStrategy",
    "AffinityPropagationStrategy",
    "HDBSCANStrategy",
    "SpectralClusteringStrategy",
    "SKLEARN_AVAILABLE",
    "HDBSCAN_AVAILABLE",
    "STRATEGY_REGISTRY",
]