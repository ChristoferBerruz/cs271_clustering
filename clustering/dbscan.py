from sklearn.base import BaseEstimator, ClassifierMixin
from attrs import define, field
import numpy as np
from typing import List
from collections import deque


@define
class DBCluster:
    points_idx: List[int] = field(factory=list)


@define
class DBSCAN(BaseEstimator, ClassifierMixin):
    m: int = field()
    epsilon: float = field()

    def fit(self, X):
        self.X_ = X
        n_samples, _ = X.shape
        self.visited = np.zeros(n_samples)
        all_points_idx = np.arange(n_samples)

        # while there are still unvisited points
        self.clusters_ = []
        while np.any(self.visited == 0):
            # pick a random point
            point_idx = np.random.choice(all_points_idx[self.visited == 0])
            self.visited[point_idx] = 1
            neigbors = self.find_neighbors(point_idx)
            if len(neigbors) < self.m:
                continue
            self.clusters_.append(DBCluster())
            cluster_idx = len(self.clusters_) - 1
            self.expand_cluster(point_idx, neigbors, cluster_idx)


    def find_neighbors(self, point_idx) -> deque:
        neighbors = deque()
        for i, sample in enumerate(self.X_):
            if np.linalg.norm(sample - self.X_[point_idx]) < self.epsilon:
                neighbors.append(i)
        return neighbors
    
    def expand_cluster(self, point_idx: int, neighbors: deque, cluster_idx: int):
        self.clusters_[cluster_idx].points_idx.append(point_idx)
        while neighbors:
            current_point_idx = neighbors.popleft()
            if self.visited[current_point_idx] == 0:
                self.visited[current_point_idx] = 1
                current_neighbors = self.find_neighbors(current_point_idx)
                if len(current_neighbors) >= self.m:
                    neighbors.extend(current_neighbors)
            for cluster in self.clusters_:
                if current_point_idx in cluster.points_idx:
                    break
            else:
                self.clusters_[cluster_idx].points_idx.append(current_point_idx)

    def predict(self, X):
        # assume the same data
        n_samples, _ = X.shape
        predictions = np.zeros(n_samples)
        for i, _ in enumerate(X):
            for j, cluster in enumerate(self.clusters_):
                if i in cluster.points_idx:
                    predictions[i] = j
                    break
            else:
                predictions[i] = -1
        return predictions