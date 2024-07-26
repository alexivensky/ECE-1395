import numpy as np


# X_train: m x (n + 1)
# y_train: m x 1
# lambda: scalar
# theta: (n + 1) x 1
def Reg_normalEqn(X_train, y_train, lamb):
    n = X_train.shape[1] - 1
    D = np.identity(n + 1)
    D[0, 0] = 0
    theta = np.linalg.pinv(X_train.T @ X_train + lamb * D) @ (X_train.T @ y_train)
    return theta


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
    print(Reg_normalEqn(toy_x, toy_y, 0.01))
