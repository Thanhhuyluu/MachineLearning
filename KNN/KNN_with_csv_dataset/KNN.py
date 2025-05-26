from collections import Counter

import numpy as np


def euclidean_distance(x1, x2):
    distance = np.sqrt(sum((x1-x2)**2))
    return distance

class KNN:
    def __init__(self, k = 3):
        self.k = k
    def fit(self, X_train, y):
        self.X_train =  np.array(X_train)
        self.y_train = np.array(y)

    def predict(self,X):
        return [self._predict(x) for x in X]
    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return Counter(k_nearest_labels).most_common()[0][0]