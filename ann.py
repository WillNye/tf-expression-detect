import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from util import sigmoid, sigmoid_cost, get_binary_data, get_data, accuracy, relu, cost, softmax, y2_indicator


class ANN:
    def __init__(self, M, use_relu):
        self.M = M
        self.use_relu = use_relu

    def forward(self, X):
        Z = relu(X.dot(self.W1) + self.b1) if self.use_relu else np.tanh(X.dot(self.W1) + self.b1)
        return sigmoid(Z.dot(self.W2) + self.b2), Z

    def predict(self, X):
        pY, _ = self.forward(X)
        return np.round(pY)

    def score(self, X, Y):
        prediction = self.predict(X)
        return accuracy(Y, prediction)


class LogisticANN(ANN):
    def __init__(self, M, use_relu=True):
        ANN.__init__(self, M, use_relu)

    def fit(self, X, Y, learning_rate=10e-7, reg=10e-7, epochs=10000, show_fig=False):
        X, Y = shuffle(X, Y)
        X_valid, Y_valid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]
        T = y2_indicator(Y)
        N, D = X.shape
        K = len(set(Y))

        self.W1 = np.random.randn(D, self.M) / np.sqrt(D + self.M)
        self.b1 = np.zeros(self.M)
        self.W2 = np.random.randn(self.M, K) / np.sqrt(self.M + K)
        self.b2 = np.zeros(K)

        costs = []
        best_validation_accuracy = 0

        for i in range(epochs):
            pY, Z = self.forward(X)

            pY_T = pY - T
            self.W2 -= learning_rate*(Z.T.dot(pY_T) + reg*self.W2)
            self.b2 -= learning_rate*(pY_T.sum(axis=0) + reg*self.b2)
            dZ = pY_T.dot(self.W2.T)
            dZ = dZ * (Z > 0) if self.use_relu else dZ * (1 - Z * Z)

            self.W1 -= learning_rate*(X.T.dot(dZ) + reg*self.W1)
            self.b1 -= learning_rate*(dZ.sum(axis=0) + reg*self.b1)

            if i%10 == 0:
                pY_valid, _ = self.forward(X_valid)
                c = cost(Y_valid, pY_valid)
                costs.append(c)
                a = accuracy(Y_valid, np.argmax(pY_valid, axis=1))

                if a > best_validation_accuracy:
                    best_validation_accuracy = a

                print('i: {}, cost: {}, accuracy: {}'.format(i, c, a))
        print('Highest level of accuracy: {}'.format(best_validation_accuracy))

        if show_fig:
            plt.plot(costs)
            plt.show()


class SigmoidANN(ANN):
    def __init__(self, M, use_relu=True):
        ANN.__init__(self, M, use_relu)

    def fit(self, X, Y, learning_rate=5*10e-7, reg=1.0, epochs=10000, show_fig=False):
        X, Y = shuffle(X, Y)
        X_valid, Y_valid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]

        N, D = X.shape
        self.W1 = np.random.randn(D, self.M) / np.sqrt(D + self.M)
        self.b1 = np.zeros(self.M)
        self.W2 = np.random.randn(self.M) / np.sqrt(self.M)
        self.b2 = 0

        costs = []
        best_validation_accuracy = 0

        for i in range(epochs):
            pY, Z = self.forward(X)

            pY_Y = pY - Y
            self.W2 -= learning_rate*(Z.T.dot(pY_Y) + reg*self.W2)
            self.b2 -= learning_rate*((pY_Y).sum() + reg*self.b2)
            dZ = np.outer(pY_Y, self.W2)
            dZ = dZ * (Z > 0) if self.use_relu else dZ * (1 - Z * Z)

            self.W1 -= learning_rate*(X.T.dot(dZ) + reg*self.W1)
            self.b1 -= learning_rate*(np.sum(dZ, axis=0) + reg*self.b1)

            if i%20 == 0:
                pY_valid, _ = self.forward(X_valid)
                c = sigmoid_cost(Y_valid, pY_valid)
                costs.append(c)
                a = accuracy(Y_valid, np.round(pY_valid))

                if a > best_validation_accuracy:
                    best_validation_accuracy = a

                print('i: {}, cost: {}, accuracy: {}'.format(i, c, a))
        print('Highest level of accuracy: {}'.format(best_validation_accuracy))

        if show_fig:
            plt.plot(costs)
            plt.show()


def main(use_sigmoid=True):
    X, Y = get_binary_data()
    X0 = X[Y == 0, :]
    X1 = X[Y == 1, :]
    X1 = np.repeat(X1, 9, axis=0)
    X = np.vstack([X0, X1])
    Y = np.array([0] * len(X0) + [1] * len(X1))

    model = SigmoidANN(100, use_relu=False) if use_sigmoid else LogisticANN(100, use_relu=False)
    model.fit(X, Y, show_fig=True)


if __name__ == '__main__':
    main()

