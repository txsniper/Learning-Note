import numpy as np

class GD(object):
    def __init__(self, learning_rate, n_iter, X_train, y_train):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.X_train = X_train
        self.y_train = y_train
    
    def batch_GD(self):
        m = self.X_train.shape[0]
        n = self.X_train.shape[1]
        theta = np.random.randn(n, 1)       
        for iter in range(self.n_iter):
            gradients = 2/m * self.X_train.T.dot(self.X_train.dot(theta) - self.y_train)
            theta = theta - self.learning_rate * gradients

    def stochastic_GD(self):
        m = self.X_train.shape[0]
        n = self.X_train.shape[1]
        n_epochs = 50
        t0, t1 = 5, 50

        def learning_schedule(t):
            return t0 / (t + t1)

        theta = np.random.randn(n, 1)
        for epoch in range(n_epochs):
            for i in range(m):
                random_index = np.random.randint(m)
                xi = self.X_train[random_index : random_index + 1]
                yi = self.y_train[random_index : random_index + 1]
                gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
                eta = learning_schedule(epoch * m + i)
                theta = theta - eta * gradients
