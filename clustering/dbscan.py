from sklearn.base import BaseEstimator, ClassifierMixin
from attrs import define, field
import numpy as np
from typing import List

@define
class DBCluster:
    core_point: np.ndarray = field()
    points_idx: List[int] = field(factory=list)


@define
class DBSCAN(BaseEstimator, ClassifierMixin):
    m: int = field()
    epsilon: float = field()

    def fit(self, X):
        self.X_ = X
        n_samples, _ = X.shape
        visited = np.zeros(n_samples)
        all_points_idx = np.arange(n_samples)

        # while there are still unvisited points
        clusters = []
        while np.any(visited == 0):
            # pick a random point
            point_idx = np.random.choice(all_points_idx[visited == 0])
            point = X[point_idx]
            visited[point_idx] = 1
            if self._is_core_point(point, X):
                cluster = DBCluster(core_point=point)
                cluster.points_idx.append(point_idx)
                self._expand_cluster(cluster, X, visited)
                clusters.append(cluster)
        # Allocate a -1 class for outliers
        self.classes_ = np.arange(start=-1, stop=len(clusters))
        self.clusters_ = clusters
        self.core_points_ = np.vstack([cluster.core_point for cluster in clusters])
        return self

    def _is_core_point(self, point: np.ndarray, X: np.ndarray) -> bool:
        # calculate the distance between the point and all other points
        distances = np.linalg.norm(X - point, axis=1)
        # count the number of points within epsilon
        n_points = np.sum(distances < self.epsilon)
        return n_points >= self.m
    
    def _expand_cluster(self, cluster: DBCluster, X: np.ndarray, visited: np.ndarray):
        # find all points within epsilon
        distances = np.linalg.norm(X - cluster.core_point, axis=1)
        reachable_points = np.argwhere(distances < self.epsilon)
        for point_idx in reachable_points:
            if visited[point_idx] == 0:
                visited[point_idx] = 1
                if self._is_core_point(X[point_idx], X):
                    cluster.points_idx.append(point_idx)
                    self._expand_cluster(cluster, X, visited)
                else:
                    # add the point to the cluster
                    cluster.points_idx.append(point_idx)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for i, point in enumerate(X):
            # calculate the distance between the point and all core points
            distances = np.linalg.norm(self.core_points_ - point, axis=1)
            # find the one that is <= epsilon
            closest_core_point = np.argmin(distances)
            if distances[closest_core_point] <= self.epsilon:
                predictions[i] = closest_core_point
            else:
                predictions[i] = -1
        return predictions