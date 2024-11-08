import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass
class WeakClassifier:
    hardcoded_labels: np.ndarray

    def __call__(self, i: int) -> int:
        return self.hardcoded_labels[i]

    @classmethod
    def all_from_data(self, data: np.ndarray, labels: np.ndarray) -> List['WeakClassifier']:
        # each column is a classified where all rows contain the lables
        classifiers = []
        for i in range(data.shape[1]):
            classifiers.append(WeakClassifier(data[:, i]))
        return classifiers


def adaboost(X: np.ndarray, z: np.ndarray, classifiers: List[WeakClassifier]):
    n_samples = len(X)
    assert len(z) == n_samples, "Number of samples must match number of labels"
    l_classifiers = len(classifiers)
    strong_classifiers = np.zeros((l_classifiers, n_samples))
    update_c_matrix = strong_classifiers
    helper_coefficient = np.zeros((1, n_samples))
    used = np.zeros(l_classifiers)
    for m in range(l_classifiers):
        if m == 0:
            m_minus_one = 0
            read_c_matrix = helper_coefficient
        else:
            m_minus_one = m - 1
            read_c_matrix = strong_classifiers
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
        for i in range(n_samples):
            new_val = read_c_matrix[m - 1, i] + alpha_m*km(X[i])
            update_c_matrix[m, i] = new_val
    # Simply return the classifiers transposed.
    return strong_classifiers.transpose()
