from __future__ import annotations

import random
import logging

# Configure logging for the example
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Import from the package
from geospatial_clustering import GeoClusterer, SKLEARN_AVAILABLE, HDBSCAN_AVAILABLE

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

        logger.info("\nRunning example with 'affinity' strategy (AffinityPropagation)...")
        # AffinityPropagation may find many clusters, or few, depending on data and preference
        # It doesn't take eps, but distance_threshold in GeoClusterer still matters for graph construction
        clusterer_affinity = GeoClusterer(
            points, resolution=9, distance_threshold=3, strategy="affinity", preference=-50 
            # preference can be tuned; a common value is the median of similarities, or a fixed scalar.
            # Negative values make points less likely to be exemplars.
            # Damping can also be tuned.
        )
        clusters_af = clusterer_affinity.clusters()
        logger.info(f"AffinityPropagation Strategy: Found {len(clusters_af)} clusters")
        for i, clus in enumerate(clusters_af):
            logger.info(f"Cluster {i}: size={len(clus)} H3s: {clus[:3]}...")

        logger.info("\nRunning example with 'spectral' strategy (SpectralClustering)...")
        # SpectralClustering needs n_clusters. For this example, let's assume we expect 2 main groups.
        # distance_threshold for graph construction is important.
        clusterer_spectral = GeoClusterer(
            points, resolution=9, distance_threshold=3, strategy="spectral", n_clusters=2
        )
        clusters_sp = clusterer_spectral.clusters()
        logger.info(f"SpectralClustering Strategy: Found {len(clusters_sp)} clusters (requested 2)")
        for i, clus in enumerate(clusters_sp):
            logger.info(f"Cluster {i}: size={len(clus)} H3s: {clus[:3]}...")

    else:
        logger.warning("\nscikit-learn not installed. Skipping DBSCAN, AffinityPropagation, and SpectralClustering strategy examples.")

    # Example for HDBSCAN if available
    # Need to import HDBSCAN_AVAILABLE from geospatial_clustering.strategies or top-level __init__
    # Assuming HDBSCAN_AVAILABLE is now imported at the top of main.py
    if HDBSCAN_AVAILABLE:
        logger.info("\nRunning example with 'hdbscan' strategy...")
        clusterer_hdbscan = GeoClusterer(
            points, resolution=9, distance_threshold=2, strategy="hdbscan", 
            min_cluster_size=5, min_samples=1 
            # min_samples=1 to ensure even small groups can be considered.
            # distance_threshold for graph is 2.
        )
        clusters_h = clusterer_hdbscan.clusters()
        logger.info(f"HDBSCAN Strategy: Found {len(clusters_h)} clusters")
        for i, clus in enumerate(clusters_h):
            logger.info(f"Cluster {i}: size={len(clus)} H3s: {clus[:3]}...")
    else:
        logger.warning("\nhdbscan library not installed. Skipping HDBSCAN strategy example.")


if __name__ == "__main__":  # pragma: no cover
    _example()