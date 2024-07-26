import numpy as np


def computeCost(X, y, theta):
    m = np.size(y, axis=0)
    J = (y - X @ theta).T @ (y - X @ theta)
    return J * (1 / (2 * m))


if __name__ == '__main__':
    x0 = np.array([1, 1, 1, 1])
    x1 = np.array([1, 2, 3, 4])
    x2 = np.array([1, 2, 3, 4])
    X = np.hstack((x0.reshape(-1, 1), x1.reshape(-1, 1), x2.reshape(-1, 1)))
    y = np.array([8, 6, 4, 2]).reshape(-1,1)
    theta1 = np.array([0, 1, 0.5]).reshape(-1,1)
    theta2 = np.array([10, -1, -1]).reshape(-1,1)
    theta3 = np.array([3.5, 0, 0]).reshape(-1,1)
    x3 = np.array([[1, 1, 1], [1, 2, 2], [1, 3, 3], [1, 4, 4]])
    print("Cost 1:", computeCost(X, y, theta1))
    print("Cost 2:", computeCost(X, y, theta2))
    print("Cost 3:", computeCost(X, y, theta3))
