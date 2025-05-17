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

## Clustering strategies

The library supports the following clustering strategies:

* `ConnectedComponentsStrategy`: Clusters are weakly-connected components in the graph.
* `DBSCANStrategy`: Clusters are based on density-based clustering.
* `HDBSCANStrategy`: Hierarchical Density-Based Spatial Clustering of Applications with Noise. It's an extension of DBSCAN that can find clusters of varying densities and is more robust to parameter selection and doesn't require the eps parameter in the same way. (Requires the hdbscan library)
* `AffinityPropagationStrategy`: Clusters are based on the Affinity Propagation algorithm. Identifies "exemplars" that represent clusters. It doesn't require the number of clusters to be specified beforehand. (Requires the scikit-learn library)
* `SpectralClusteringStrategy`: Clusters are based on the Spectral Clustering algorithm. This method uses the eigenvalues of a similarity matrix to perform dimensionality reduction before clustering(Requires the scikit-learn library)
* `MeanShiftStrategy`: Clusters are based on the Mean Shift algorithm. Mean shift clustering aims to discover “blobs” in a smooth density of samples. It is a centroid-based algorithm, which works by updating candidates for centroids to be the mean of the points within a given region. These candidates are then filtered in a post-processing stage to eliminate near-duplicates to form the final set of centroids. (Requires the scikit-learn library)
* `AgglomerativeClusteringStrategy`: Clusters are based on the Agglomerative Clustering algorithm. (Requires the scikit-learn library)
* `LouvainStrategy`: Clusters are based on the Louvain algorithm. A greedy optimization method for community detection to extract non-overlapping communities from large networks. (Requires the python-louvain library)
* `LeidenStrategy`: Clusters are based on the Leiden algorithm (a modification of the Louvain method). Like the Louvain method, it attempts to optimize modularity in extracting communities from networks, but it addresses key issues present in the Louvain method, namely poorly connected communities and the resolution limit of modularity. (Requires the leidenalg library)


**Next steps:**

* Try tuning the H3 resolution—smaller hexes bring finer spatial detail.
* Drop in alternative strategies (e.g., community detection) by subclassing `ClusteringStrategy`.
* Scale out: the `_build_graph` method is isolated—swap in a parallel implementation or sparse backend if you hit performance ceilings.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.