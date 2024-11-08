import numpy as np
from typing import List, Callable, Optional
from dataclasses import dataclass, field
import random


@dataclass
class WeakClassifier:
    classes: List[int]
    probs: Optional[List[float]] = None

    def __post_init__(self):
        if self.probs is None:
            self.probs = [
                1/len(self.classes) + random.uniform(0, 1) for _ in range(len(self.classes))]
            # normalize the probabilities
            prob_sum = sum(self.probs)
            self.probs = [p/prob_sum for p in self.probs]

    def __call__(self, value: np.ndarray):
        return random.choice(self.classes, self.probs)


def adaboost(X: np.ndarray, z: np.ndarray, classifiers: List[Callable[[np.ndarray], int]]):
    n_samples, m_features = X.shape
    assert len(z) == n_samples, "Number of samples must match number of labels"
    l_classifiers = len(classifiers)
    coefficient_matrix = np.zeros((l_classifiers, n_samples))
    update_c_matrix = coefficient_matrix
    helper_coefficient = np.zeros((1, n_samples))
    used = np.zeros(l_classifiers)
    for m in range(l_classifiers):
        if m == 0:
            m_minus_one = 0
            read_c_matrix = helper_coefficient
        else:
            m_minus_one = m - 1
            read_c_matrix = coefficient_matrix
        weights = np.zeros(n_samples)
        for i in range(n_samples):
            weights[i] = np.exp ** (-z[i]*read_c_matrix[m_minus_one, i])
        W = np.sum(weights)
        W2 = np.inf
        t = None
        for j in range(l_classifiers):
            if used[j] == 0:
                predictions = [classifiers[j](X[i]) for i in range(n_samples)]
                Y = np.sum(np.where(predictions == z, weights))
                if Y < W2:
                    W2 = Y
                    t = j
        km = classifiers[t]
        used[t] = 1
        rm = W2 // W
        alpha_m = 0.5*np.log((1 - rm)/rm)
        new_val = read_c_matrix[m - 1, i] + alpha_m*km(X[i])
        update_c_matrix[m, i] = new_val
