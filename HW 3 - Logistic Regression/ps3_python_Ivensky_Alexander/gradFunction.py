import numpy as np
from sigmoid import *

# X_train: m x (n + 1)
# y_train: m x 1
# theta: (n + 1) x 1
def gradFunction(theta, X_train, y_train):
    m = X_train.shape[0] # number of training samples
    n = X_train.shape[1] - 1 # number of non-bias features
    grad = np.zeros((n + 1))
    
    for j in range(n + 1):
        for i in range(m):
            z = theta.T @ X_train[i, :]
            h = sigmoid(z)
            grad[j] += (h - y_train[i, 0]) * X_train[i, j]

    return grad

if __name__ == "__main__":
    toy_x = np.array([[1, 1, 0],
                     [1, 1, 3],
                     [1, 3, 1],
                     [1, 3, 4]])
    toy_y = np.array([[0],
                     [1],
                     [0],
                     [1]])
    toy_theta = np.array([[2],
                          [0],
                          [0]])
    print(gradFunction(toy_theta, toy_x, toy_y))
