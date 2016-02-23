Caleydo Clustering
==================

This repository is a server-side plugin for Caleydo to apply clustering algorithms on arbitrary matrices.

Supported Algorithms:
--------------------
- K-Means (Init methods: Forgy, Uniform, Random, Kmeans++)
- Hierarchical Clustering (Single, Complete, Weighted, Median)
- Affinity Propagation

General Information:
-------------------
- All these algorithms expect the input as matrix (or 2D numpy array)
- It is assumed that the matrix is dense (no sparse matrix support right now)
- NaN values will be converted to zero
