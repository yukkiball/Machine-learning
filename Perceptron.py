import numpy as np

class Perceptron:

    """感知机的python实现"""

    def __init__(self):
        self.theta_ = None
        self.coef_ = None
        self.inter_ = None

    def fit(self, X_train, y_train, eta=1, n_iters=7):
        """使用随机梯度下降法进行拟合"""

        def J(theta, X_b, y):
            try:
                return -y.T.dot(X_b.dot(theta))
            except:
                return 0.0

        def dJ(X_b, y):
            try:
                return -y.t.dot(X_b)
            except:
                return 0.0

        def gradient_descent(X_b, y, initial_theta, eta, n_iters, epsilon=1e-8):

            theta = initial_theta
            i_iter = 0
            while i_iter < n_iters:
                indexes = np.random.permutation(len(X_b))
                X_b_new = X_b[indexes]
                y_new = y[indexes]
                for i in range(len(X_b)):
                    if y_new[i] * X_b_new[i].T.dot(theta) <= 0:
                        gradient = -y_new[i] * X_b_new[i]
                        break
                last_theta = theta.copy()
                theta -= eta * gradient
                if abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon:
                    break
                i_iter += 1
            # index = [0, 2, 2, 2, 0, 2, 2]
            # while i_iter < n_iters:
            #     gradient = -y[index[i_iter]] * X_b[index[i_iter]]
            #     theta -= eta * gradient
            #     i_iter += 1
            return theta

        X_b = np.hstack([np.ones([len(X_train), 1]), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self.theta_ = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)
        self.inter_ = self.theta_[0]
        self.coef_ = self.theta_[1:]

        return self

    def predict(self, X_predict):

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self.theta_)

    def __repr__(self):
        return "Perceptron()"

p1 = Perceptron()
X = np.array([
    [3, 3],
    [4, 3],
    [1, 1]
])
y = np.array([1, 1, -1])
p1.fit(X, y)
print(p1.coef_)
print(p1.inter_)




