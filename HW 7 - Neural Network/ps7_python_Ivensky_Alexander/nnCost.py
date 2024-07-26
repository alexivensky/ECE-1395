import numpy as np
from scipy.io import loadmat
from predict import *

def nnCost(Theta1, Theta2, X, y, K, lda):
    # prediction
    p, h_x = predict(Theta1, Theta2, X) 
    # transforming y to NN encoding
    y_k = np.zeros((y.shape[0], K))
    for i in range(y.shape[0]):
        y_k[i, (y[i, 0] - 1)] = 1 
    # first term of cost function
    J = 0
    for i in range(X.shape[0]):
        for k in range(K):
            J += y_k[i, k] * np.log(h_x[i, k] + 1e-8) + (1 - y_k[i, k]) * np.log(1 - h_x[i, k] + 1e-8)
    J = J * (-1 / X.shape[0])
    # second term
    reg = np.sum(Theta1 ** 2) + np.sum(Theta2 ** 2)
    reg = reg * (lda / (2 * X.shape[0]))
    return (J + reg)
        
    

if __name__ == "__main__":
    data = loadmat('input/HW7_Data2_full.mat') 
    X = data['X']
    weights = loadmat('input/HW7_weights_3_full.mat')
    Theta1 = weights['Theta1']
    Theta2 = weights['Theta2']
    y_labels = data['y_labels']
    cost = nnCost(Theta1, Theta2, X, y_labels, 3, 0.1)
    print(cost)
    