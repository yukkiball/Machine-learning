import numpy as np
from collections import Counter
from math import sqrt

class KNNclassify:
    """k邻近分类器"""
    def __init__(self, k):
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        y_predict = [self._predict(x) for x in X_predict]
        return y_predict

    def _predict(self, x):
        distances = [sqrt(np.sum((x - x_train)**2)) for x_train in self._X_train]
        nearest = np.argsort(distances)

        top_y = [self._y_train[i] for i in nearest[:self.k]]
        num = Counter(top_y)
        return num.most_common(1)[0][0]

    def __repr__(self):
        return "KNNclassify(k = {})".format(self.k)

knn = KNNclassify(1)
X = np.array([
    [3, 3],
    [4, 3],
    [1, 1]
])
y = np.array([1, 2, 3])
knn.fit(X, y)
X_predict = np.array([
    [2, 2],
    [3, 3],
    [4, 4]
]
)
print(knn)
print(knn.predict(X_predict))

