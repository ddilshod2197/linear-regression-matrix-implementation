import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.w) + self.b

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        y_approximated = np.dot(X, self.w) + self.b
        return y_approximated

# Masalan
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 3, 5, 7, 11])

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(np.array([[6]]))
print(y_pred)
