import numpy as np

class SVMSMO:
    """使用SMO实现SVM(二分类)"""
    def __init__(self, max_iters=10000, kernel_type='gaussian', C=1.0, epsilon=0.001, sigma=5.0):
        self.kernels = {
            'poly': self.kernel_poly,
            'gaussian': self.kernel_gaussian
        }
        self.max_iters = max_iters
        self.kernel_type = kernel_type
        self.kernel_func = self.kernels[kernel_type]
        self.epsilon = epsilon
        self.sigma = sigma
        self.alpha = None
        self.C = C
        self.b = 0
        self.X_train = None
        self.w = None

    #使用SMO计算alpha和b
    def fit(self, X, y):
        self.X_train = X.copy()
        m, n = X.shape
        self.alpha = np.zeros(m)
        count = 0
        while True:
            count += 1
            alpha_old = self.alpha.copy()   #存储老的alpha
            for i in range(m):
                j = self.get_j(0, m - 1, i)
                alpha_old_i = self.alpha[i]
                alpha_old_j = self.alpha[j]
                H, L = self.cal_L_H(y[i], y[j], alpha_old_i, alpha_old_j)   #计算剪辑边界
                E_i = self.E(X, y, i)   #计算误差
                E_j = self.E(X, y, j)
                eta = self.kernel_func(X[i], X[i]) + self.kernel_func(X[j], X[j]) - 2* self.kernel_func(X[i], X[j])   #计算eta
                #未剪辑时候alpha_j的解
                self.alpha[j] += y[j] * (E_i - E_j) / eta
                #剪辑
                if self.alpha[j] > H:
                    self.alpha[j] = H
                if self.alpha[j] < L:
                    self.alpha[j] = L
                #更新alpha_i的解
                self.alpha[i] = self.alpha[j] + y[i] * y[j] * (alpha_old_j - self.alpha[j])
                #更新b
                b_i = self.b - E_i - y[i] * self.kernel_func(X[i], X[i]) * (self.alpha[i] - alpha_old_i) - \
                    y[j] * self.kernel_func(X[j], X[i]) * (self.alpha[j] - alpha_old_j)
                b_j = self.b - E_j - y[i] * self.kernel_func(X[i], X[j]) * (self.alpha[i] - alpha_old_i) - \
                      y[j] * self.kernel_func(X[j], X[j]) * (self.alpha[j] - alpha_old_j)
                if self.alpha[i] > 0 and self.alpha[i] < self.C:
                    self.b = b_i
                elif self.alpha[j] > 0 and self.alpha[j] < self.C:
                    self.b = b_j
                else:
                    self.b = (b_i + b_j) / 2
            self.w = self.alpha * y
            if np.linalg.norm(self.alpha - alpha_old) < self.epsilon:
                break
            if count >= self.max_iters:
                break
            return self

    #预测
    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        sum = np.sum([self.alpha[j] * y[j] * self.kernel_func(X[j], x) for j in range(len(self.X_train))])
        return np.sign(sum + self.b)

    #多项式核函数
    def kernel_poly(self, x1, x2):
        return np.dot(x1, x2.T)

    #高斯核函数
    def kernel_gaussian(self, x1, x2, sigma=5.0):
        if self.sigma:
            sigma = self.sigma
        return np.exp(-np.linalg.norm(x1-x2)**2 / (2 * (sigma ** 2)))

    #获取不等于i的j
    def get_j(self, a, b, i):
        j = i
        while j == i:
            j = np.random.randint(a, b)
        return j

    #计算剪辑的边界
    def cal_L_H(self, y_i, y_j, alpha_old_i, alpha_old_j):
        if y_i != y_j:
            L = max(0, alpha_old_j - alpha_old_i)
            H = min(self.C, self.C + alpha_old_j - alpha_old_i)
        else:
            L = max(0, alpha_old_i + alpha_old_j - self.C)
            H = min(self.C, alpha_old_i + alpha_old_j)
        return H, L

    #计算误差
    def E(self, X, y, i):
        sum = np.sum([self.alpha[j] * y[j] * self.kernel_func(X[j], X[i]) for j in range(len(X))])
        return sum + self.b - y[i]

    def __repr__(self):
        return  "SVM(kernal = {})".format(self.kernel_type)

