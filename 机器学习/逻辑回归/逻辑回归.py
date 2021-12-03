import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as opt


def getDate():
    path = "ex2data1.txt"

    data = pd.read_csv(path, names=['exam1', 'exam2', 'admitted'])

    col = data.shape[1]

    X = data.iloc[:, 0:col - 1]
    y = data.iloc[:, col - 1:col]

    X = np.matrix(X.values)
    y = np.matrix(y.values)
    data = np.matrix(data.values)

    return data, X, y


def handleData(X):
    X = np.insert(X, 0, values=1, axis=1)
    return X


def plotData(X, y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)

    pos = np.matrix(pos)
    neg = np.matrix(neg)

    print(X[pos, 1])

    plt.plot(X[pos, 0], X[pos, 1], 'ko', MarkerFaceColor='r', MarkerEdgeColor='r', MarkerSize=6)
    plt.plot(X[neg, 0], X[neg, 1], 'ko', MarkerFaceColor='b', MarkerEdgeColor='b', MarkerSize=6)
    plt.show()


def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


def featureScaling(data):
    data = (data - data.mean()) / data.std()
    return data


def costFunction(theta, X, y):
    # m = len(y)
    # z = X * theta.T
    # error = -y.T * (np.log(sigmoid(z))) - (1 - y).T * (np.log(1 - sigmoid(z)))
    # j = np.sum(error) / m
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))


def gradient(theta, X, y):
    grad = (1 / len(y)) * (sigmoid(X * theta.T) - y).T * X
    return grad


def gradientDecent(theta, X, y, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    cost_list = np.zeros(iters)
    m = y.shape[0]
    for i in range(iters):
        error = sigmoid(X * theta) - y
        for j in range(theta.shape[0]):
            term = np.multiply(error, X[:, j])
            temp[j] = theta[j] - (alpha / m) * np.sum(term)
        theta = temp
        cost_list[i] = costFunction(theta, X, y)

    return cost_list, theta


def predict(X, theta):
    prob = sigmoid(X * theta.T)
    return (prob >= 0.5).astype(int)


if __name__ == '__main__':
    data, X, y = getDate()
    # data = featureScaling(data)
    X = handleData(X)
    # plotData(X[:, 1:3], y)
    # theta = np.array([[0], [0], [0]])
    theta = np.zeros(3)

    # print(J)
    # cost, theta = gradientDecent(theta, X, y, 0.0012, 1000000)

    # print(theta)
    # print(cost)

    # plt.plot(np.arange(1000000), cost, 'r')

    J = costFunction(theta, X, y)
    grad = gradient(theta, X, y)
    print(J)
    print(grad)

    # res = opt.minimize(fun=costFunction, x0=theta, args=(X, y), method='Newton-CG', jac=gradient)
    # print(res)

    # plt.plot(np.arange(-10, 10, step=0.01), sigmoid(np.arange(-10, 10, step=0.01)))
    # plt.show()