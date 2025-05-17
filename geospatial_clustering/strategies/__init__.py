from __future__ import annotations

from .base_strategy import ClusteringStrategy
from .connected_components_strategy import ConnectedComponentsStrategy
from .dbscan_strategy import DBSCANStrategy, SKLEARN_AVAILABLE # Expose SKLEARN_AVAILABLE if needed elsewhere

# Registry for convenience
STRATEGY_REGISTRY = {
    "components": ConnectedComponentsStrategy,
    "dbscan": DBSCANStrategy,
}

__all__ = [
    "ClusteringStrategy",
    "ConnectedComponentsStrategy",
    "DBSCANStrategy",
    "SKLEARN_AVAILABLE", # If other parts of the library might need to check this
    "STRATEGY_REGISTRY",
]