from __future__ import annotations

import random
import logging

# Configure logging for the example
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Import from the package
from geospatial_clustering import GeoClusterer, SKLEARN_AVAILABLE

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

    logger.info("Running example with 'components' strategy...")
    clusterer_components = GeoClusterer(
        points, resolution=9, distance_threshold=1, strategy="components"
    )
    clusters_c = clusterer_components.clusters()
    logger.info(f"Components Strategy: Found {len(clusters_c)} clusters")
    for i, clus in enumerate(clusters_c):
        logger.info(f"Cluster {i}: size={len(clus)} H3s: {clus[:3]}...") # Print first 3 H3s

    # Example of labelled points
    logger.info("\nLabelled points (Components):")
    count = 0
    for lat, lon, label in clusterer_components.labelled_points():
        if count < 5: # Print first 5
            logger.info(f"Point ({lat:.3f}, {lon:.3f}) -> Label {label}")
        count += 1


    if SKLEARN_AVAILABLE:
        logger.info("\nRunning example with 'dbscan' strategy...")
        clusterer_dbscan = GeoClusterer(
            points, resolution=9, distance_threshold=2, strategy="dbscan", min_samples=2
        )
        # Note: DBSCAN eps should align with distance_threshold used for graph.
        # Here, distance_threshold for graph is 2, and eps for DBSCAN is also 2 (implicitly passed).
        clusters_d = clusterer_dbscan.clusters()
        logger.info(f"DBSCAN Strategy: Found {len(clusters_d)} clusters")
        for i, clus in enumerate(clusters_d):
            logger.info(f"Cluster {i}: size={len(clus)} H3s: {clus[:3]}...")

        logger.info("\nLabelled points (DBSCAN):")
        count = 0
        for lat, lon, label in clusterer_dbscan.labelled_points():
            if count < 5: # Print first 5
                logger.info(f"Point ({lat:.3f}, {lon:.3f}) -> Label {label}")
            count += 1
    else:
        logger.warning("\nscikit-learn not installed. Skipping DBSCAN strategy example.")


if __name__ == "__main__":  # pragma: no cover
    _example()