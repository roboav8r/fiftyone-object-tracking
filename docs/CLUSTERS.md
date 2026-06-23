# Clusters

The **Clusters** tab clusters trajectory *shapes* and surfaces them as an
interactive dendrogram. Clustering is computed on demand by the
`cluster_trajectories` operator: pairwise Dynamic Time Warping (DTW)
distances (`_dtw.py`) feed hierarchical clustering (`_clustering.py`),
and the result is drawn as a dendrogram with a drag-to-cut threshold and
click-a-cluster-to-select interaction.

> Source dataset schema: see [DATASET_SCHEMA.md](DATASET_SCHEMA.md).

## Cluster trajectories

On the trajectories dataset, open the `ObjectTracking` panel and select
the **Clusters** tab. Run clustering, then drag the dendrogram threshold
to cut the tree at a chosen height and click a cluster to select its
member trajectories.
