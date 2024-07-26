import numpy as np
import math
from sigmoid import *

# X_train: m x (n + 1)
# y_train: m x 1
# theta: (n + 1) x 1
def costFunction(theta, X_train, y_train):
    J = 0
    m = X_train.shape[0]
    for i in range(m):
        z = theta.T @ X_train[i, :]
        h = sigmoid(z)
        J += -1 * y_train[i, 0] * math.log(h + 1e-15) - (1 - y_train[i, 0]) * math.log(1 - h + 1e-15)
    return J * (1 / m)


if __name__ == "__main__":
    toy_x = np.array([[1, 1, 0],
                     [1, 1, 3],
                     [1, 3, 1],
                     [1, 3, 4]])
    toy_y = np.array([[0],
                     [1],
                     [0],
                     [1]])
    toy_theta = np.array([[2],[0],[0]])
    print("Toy set cost:", costFunction(toy_theta, toy_x, toy_y))