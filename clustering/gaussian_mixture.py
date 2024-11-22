import numpy as np
from typing import List

from attrs import define, field

from sklearn.base import BaseEstimator, ClassifierMixin

@define
class Theta:
    u: np.ndarray = field()
    S: np.ndarray = field()

@define
class EMCluster:
    theta: Theta = field()
    tau: float = field()


@define
class GaussianMixture:
    clusters: List[EMCluster] = field()
    n_clusters: int = field(init=False)
    k_features: int = field(init=False)

    def __attrs_post_init__(self):
        self.n_clusters = len(self.clusters)
        self.k_features = self.clusters[0].theta.u.shape[0]

    def e_step(self, data: np.ndarray) -> np.ndarray:
        n_samples = data.shape[0]
        pji = np.zeros((self.n_clusters, n_samples))
        for j in range(self.n_clusters):
            for i in range(n_samples):
                numerator = self.clusters[j].tau*self.probability_function_2d(data[i], self.clusters[j].theta)
                denom = 0
                for k in range(self.n_clusters):
                    denom += self.clusters[k].tau*self.probability_function_2d(data[i], self.clusters[k].theta)
                pji[j, i] = numerator/denom
        return pji
    
    def recompute_mixture_parameters(self, pji: np.ndarray, data: np.ndarray) -> np.ndarray:
        n_samples = data.shape[0]
        mixture_parameters = np.zeros(self.n_clusters)
        for j in range(self.n_clusters):
            mixture_parameters[j] = np.sum(pji[j, :])/n_samples
        return mixture_parameters
    
    def recompute_means_of_clusters(self, pji: np.ndarray, data: np.ndarray) -> np.ndarray:
        n_samples = data.shape[0]
        means = np.zeros((self.n_clusters, self.k_features))
        for j in range(self.n_clusters):
            numerator = np.zeros(self.k_features)
            denom = 0
            for i in range(n_samples):
                numerator += pji[j, i]*data[i]
                denom += pji[j, i]
            means[j] = numerator/denom
        return means
    
    def recompute_covariance_matrices(self, pji: np.ndarray, data: np.ndarray, means: np.ndarray) -> List[np.ndarray]:
        n_samples = data.shape[0]
        S = []
        for j in range(self.n_clusters):
            S_j = np.zeros((self.k_features, self.k_features))
            for i in range(n_samples):
                S_j += pji[j, i]*np.outer(data[i] - means[j], data[i] - means[j])
            S_j /= np.sum(pji[j, :])
            S.append(S_j)
        return S
    
    @staticmethod
    def probability_function_2d(vec: np.ndarray, theta: Theta) -> np.ndarray:
        u = theta.u
        S = theta.S
        row_vec = vec.reshape(-1, 1)
        row_u = u.reshape(-1, 1)
        S_inv = np.linalg.inv(S)
        numerator = np.exp(-0.5*(row_vec - row_u).transpose()
                        @ S_inv@(row_vec - row_u))
        denom = np.sqrt((2*np.pi)*np.linalg.det(S))
        return numerator/denom