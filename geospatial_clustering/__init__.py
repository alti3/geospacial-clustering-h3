from __future__ import annotations

import logging

from .geoclusterer import GeoClusterer
from .strategies import (
    ClusteringStrategy,
    ConnectedComponentsStrategy,
    DBSCANStrategy,
    AffinityPropagationStrategy,
    HDBSCANStrategy,
    SpectralClusteringStrategy,
    STRATEGY_REGISTRY,
    SKLEARN_AVAILABLE,
    HDBSCAN_AVAILABLE
)

# Configure a default null handler for the library's root logger
# to prevent "No handler found" warnings if the user of the library
# doesn't configure logging.
logging.getLogger(__name__).addHandler(logging.NullHandler())


__all__ = [
    "GeoClusterer",
    "ClusteringStrategy",
    "ConnectedComponentsStrategy",
    "DBSCANStrategy",
    "AffinityPropagationStrategy",
    "HDBSCANStrategy",
    "SpectralClusteringStrategy",
    "STRATEGY_REGISTRY",
    "SKLEARN_AVAILABLE",
    "HDBSCAN_AVAILABLE",
]

# You can also define __version__ here
# __version__ = "0.1.0"