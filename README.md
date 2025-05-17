# Geospatial Clustering with H3

## Key features & design choices

| Aspect                 | What it does                                                                                                                        | Why it matters                                                                  |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **H3 discretisation**  | Converts lat/lon to hex indexes at any resolution you pass.                                                                         | Gives you a true equal-area spatial grid, perfect for distance‐only clustering. |
| **Graph construction** | Builds a NetworkX graph using `h3_distance` ≤ *distance\_threshold*; avoids O(N²) by exploring *k-rings*.                           | Fast enough for tens-of-thousands of points while staying pure-Python/NumPy.    |
| **Strategy pattern**   | Pluggable `ClusteringStrategy` base class. Included: **ConnectedComponents** (zero deps) and **DBSCAN** (if you have scikit-learn). | Swap in Louvain, Leiden, HDBSCAN, etc., just by writing a new strategy.         |
| **NumPy everywhere**   | Coordinates and distance matrices handled as ndarrays.                                                                              | Memory-efficient and vectorised where possible.                                 |
| **Slim public API**    | `GeoClusterer(...).clusters()` and `GeoClusterer(...).labelled_points()`                                                            | Easy to integrate in ETL pipelines or notebooks.                                |
| **Repro ready**        | Example CLI section shows end-to-end use.                                                                                           | Run `python geospatial_clustering_h3.py` to see it in action.                   |

## How to use

The `main.py` script shows how to use the library.
```bash
uv sync # install dependencies
uv run main.py # runs the built-in demo
```

Example output:
```bash
INFO:__main__:Running example with 'dbscan' strategy...
INFO:__main__:DBSCAN Strategy: Found 10 clusters
INFO:__main__:Cluster 0: size=10 H3s: ['87283082bffffff', '87283082bffffff', '87283082bffffff', '87283082bffffff', '87283082bffffff', '87283082bffffff', '87283082bffffff', '87283082bffffff', '87283082bffffff', '87283082bffffff']
INFO:__main__:Cluster 1: size=10 H3s: ['87283082bffffff', '87283082bffffff', '87283082bffffff', '87283082bffffff', '87283082bffffff', '87283082bffffff', '87283082bffffff', '87283082bffffff', '87283082bffffff', '87283082bffffff']
```



Or from a notebook / script:

```python
from geospatial_clustering_h3 import GeoClusterer

clusterer = GeoClusterer(
    coords,               # list/array of (lat, lon)
    resolution=9,         # H3 resolution
    distance_threshold=1, # hex-grid steps
    strategy="dbscan",    # or "components"
    min_samples=3         # DBSCAN kwargs if relevant
)
clusters = clusterer.clusters()
```

Feel free to tweak the distance threshold, resolution, or plug in another strategy class.

**Next steps:**

* Try tuning the H3 resolution—smaller hexes bring finer spatial detail.
* Drop in alternative strategies (e.g., community detection) by subclassing `ClusteringStrategy`.
* Scale out: the `_build_graph` method is isolated—swap in a parallel implementation or sparse backend if you hit performance ceilings.
