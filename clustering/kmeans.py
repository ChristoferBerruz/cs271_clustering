from sklearn.base import ClassifierMixin, BaseEstimator
import numpy as np

from attrs import define, field
from typing import List

@define
class Cluster:
    centroid: np.ndarray
    points_idx: List[int] = field(factory=list)


class KMeans(BaseEstimator, ClassifierMixin):
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.X_ = None
        self.clusters_ = None

    def fit(self, X):
        self.X_ = X
        self.clusters_: List[Cluster] = [None]*self.n_clusters
        self.classes_ = np.arange(self.n_clusters)
        # randomly pick n_cluster centroids
        self.m, self.n = X.shape
        # initial partition into clusters
        choices = np.arange(self.m)
        centroid_indexes = np.random.choice(choices, self.n_clusters, replace=False)
        for cluster_n, i in enumerate(centroid_indexes):
            self.clusters_[cluster_n] = Cluster(centroid=X[i])

        self.compute_clusters_based_on_centroids()
        for _ in range(self.max_iter):
            stop = self.update_centroids()
            # The stop flag only tells us that the new centroids are the same
            # but we still have to recompute the clusters based on the new centroids
            # given that we clear them in the update_centroids method
            self.compute_clusters_based_on_centroids()
            if stop:
                break

        return self

    def find_closest_cluster(self, sample) -> int:
        distances = np.zeros(self.n_clusters)
        for i in range(self.n_clusters):
            distances[i] = np.linalg.norm(sample - self.clusters_[i].centroid)
        return np.argmin(distances)

    def compute_clusters_based_on_centroids(self):
        for i, sample in enumerate(self.X_):
            closest_cluster = self.find_closest_cluster(sample)
            self.clusters_[closest_cluster].points_idx.append(i)
        
    def update_centroids(self) -> bool:
        """Updates the centroids of the clusters

        Returns:
            bool: whether to stop the algorithm or not
        """
        stop = True
        for i in range(self.n_clusters):
            cluster = self.clusters_[i]
            if len(cluster.points_idx) == 0:
                continue
            new_centroid = np.mean(self.X_[cluster.points_idx], axis=0)
            if not np.allclose(cluster.centroid, new_centroid):
                # If a new centroid is found, we need to continue the algorihthm
                stop = False
            self.clusters_[i] = Cluster(centroid=new_centroid)
        return stop

    def predict(self, X) -> int:
        if self.X_ is None:
            raise ValueError("Model not yet fitted")
        
        predictions = np.zeros(len(X))
        for i, sample in enumerate(X):
            # find the closest cluster
            closest_cluster = self.find_closest_cluster(sample)
            predictions[i] = closest_cluster
        return predictions