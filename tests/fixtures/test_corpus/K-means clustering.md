# K-means clustering

Source: chat GPT4 

K-means clustering is a popular unsupervised machine learning algorithm used for partitioning a dataset into a set of k groups (where k is a predefined number). The goal is to partition the data points into groups based on attribute similarity. Here's a detailed explanation, including the key steps involved, its applications, and limitations.

### Key Steps in K-means Clustering

1. **Initialization**: Select k initial centroids randomly from the data points. These centroids are the initial centers of the clusters.

2. **Assignment**: Assign each data point to the nearest centroid, based on the distance between the data point and the centroid. This step partitions the data into clusters based on the current centroid positions. The most common distance metric used is Euclidean distance, though other metrics can be used depending on the problem domain.

3. **Update**: Recalculate the centroids of the clusters by taking the mean of all points assigned to each cluster. The centroid of a cluster is a point (not necessarily one of the input data points) that minimizes the sum of distances to the points in that cluster.

4. **Iterate**: Repeat the assignment and update steps until the centroids no longer change significantly. This indicates that the algorithm has converged to a solution.

### 