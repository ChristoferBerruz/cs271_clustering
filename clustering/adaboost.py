import numpy as np


def adaboost(X: np.ndarray, z: np.ndarray, classifiers: np.ndarray) -> np.ndarray:
    n_samples = len(X)
    assert len(z) == n_samples, "Number of samples must match number of labels"
    l_classifiers = len(classifiers)
    strong_classifiers = np.zeros((l_classifiers, n_samples))
    update_c_matrix = strong_classifiers
    helper_coefficient = np.zeros((1, n_samples))
    used = np.zeros(l_classifiers)
    for m in range(l_classifiers):
        print(f"Finding strong classifiers m = {m}")
        if m == 0:
            m_minus_one = 0
            read_c_matrix = helper_coefficient
        else:
            m_minus_one = m - 1
            read_c_matrix = strong_classifiers
        weights = np.zeros(n_samples)
        for i in range(n_samples):
            weights[i] = np.e ** (-z[i]*read_c_matrix[m_minus_one, i])
        W = np.sum(weights)
        W2 = np.inf
        t = None
        for j in range(l_classifiers):
            if used[j] == 0:
                predictions = classifiers[:, j]
                # only select the weights where the prediction is correct
                Y = np.sum([weights[i]
                           for i in range(n_samples) if predictions[i] != z[i]])
                if Y < W2:
                    W2 = Y
                    t = j
        used[t] = 1
        rm = W2 / W
        alpha_m = 0.5*np.log((1 - rm)/rm)
        update_c_matrix[m, :] = read_c_matrix[m_minus_one,
                                              :] + alpha_m*classifiers[:, t]
    # Simply return the classifiers transposed.
    return strong_classifiers.transpose()
