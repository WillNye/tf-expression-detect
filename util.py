import numpy as np
import pandas as pd

"""

I like to throw this little set of notes I made into every ML project, it's a nice little reference guide.

#### TRAINING DATA ####
X = Training Inputs -> X is of shape N*D
Y = Training Targets can also be T iff p represented as Y
    Y is of shape N*1
    Y = A column vector (2-D object)
        Y can also be a vector of only 1-D of length N
N = # of Samples


#### SIZES ####
D = # of Input Features
M = # of hidden units
K = # of output classes
D -1-N-> M -1-N-> K


#### PREDICTIONS ####
P = Predictions
Predictions = p(Y | X) -> p_y_given_x or py_x
aka P of Y given X meaning prediction of the Y value given an X value across a matrices
p(Y = K | X) is the probability of Y given a single value in X


#### WEIGHTS ####
W1 = D*M -> Input to hidden weight matrix
b1 = M -> Hidden bias
W2 = M*K -> Hidden to output bias
b2 = K -> Output bias


#### INDEXING (like loops you derp) ####
i = 1...D (input layer)
j = 1...M (hidden layer)
k = 1...k (output layer)


#### Cost/ Objective/ Error ####
E = Error


#### Likelihood ####
L = Likelihood

CAPS -> matrices
lower -> vector
gradient descent <-> backward propagation
"""


def init_w_and_b(input_size, output_size):
    W = np.random.randn(input_size, output_size) / np.sqrt(input_size + output_size)
    b = np.zeros(output_size)

    return W.astype(np.float32), b.astype(np.float32)


def init_filter(shape, pool_size):
    w = np.random.randn(*shape) / np.sqrt(np.prod(shape[1:]) + shape[0] * np.prod(shape[2:] / np.prod(pool_size)))

    return w.astype(np.float32)


def relu(x):
    return x * (x > 0)


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def sigmoid_cost(T, Y):
    return -(T*np.log(Y) + (1 - T) * np.log(1 - Y)).sum()


def softmax(a):
    exp_a = np.exp(a)
    return exp_a / exp_a.sum(axis=1, keepdims=True)


def cost(T, Y):
    return -np.log(Y[np.arange(len(T)), T]).sum()


def y2_indicator(y):
    N = len(y)
    K = len(set(y))
    T = np.zeros((N, K))
    for i in range(N):
        T[i, y[i]] = 1

    return T


def accuracy(targets, predictions):

    return 1 - np.mean(targets != predictions)


def get_data(balance_ones=False):
    Y = []
    X = []
    first = True

    for line in open('data/fer2013.csv'):
        if first:
            first = False
            continue
        row = line.split(',')
        Y.append(int(row[0]))
        X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)

    if balance_ones:
        X0, Y0 = X[Y != 1, :], Y[Y != 1]
        X1 = np.repeat(X[Y==1, :], 9, axis=0)
        X = np.vstack([X0, X1])
        Y = np.concatenate((Y0, [1]*len(X1)))
    return X, Y


def get_img_data():
    X, Y = get_data()
    N, D = X.shape
    d = int(np.sqrt(D))
    X = X.reshape(N, 1, d, d)
    return X, Y


def get_binary_data():
    Y = []
    X = []
    first = True

    for line in open('data/fer2013.csv'):
        if first:
            first = False
            continue
        row = line.split(',')
        y = int(row[0])
        if y in [0, 1]:
            Y.append(y)
            X.append([int(p) for p in row[1].split()])

    return np.array(X) / 255.0, np.array(Y)








