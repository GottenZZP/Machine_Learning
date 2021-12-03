import numpy as np


def costFunctionJ(X, y, theta):

    m = np.size(X, 1) + 1
    print(m)
    predictions = np.dot(X, theta)
    print(predictions)
    sq_errors = (predictions - y) ** 2
    print(sq_errors)
    j = 1 / (2 * m) * np.sum(sq_errors)

    return j


if __name__ == "__main__":
    X = np.array([[1, 1],
                  [1, 2],
                  [1, 3]])

    y = np.array([[1],
                  [2],
                  [3]])

    theta = np.array([[0],
                      [1]])

    J = costFunctionJ(X, y, theta)

    print(J)
