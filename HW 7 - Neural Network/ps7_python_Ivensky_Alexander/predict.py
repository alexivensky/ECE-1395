import numpy as np
from scipy.io import loadmat

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

def predict(Theta1, Theta2, X):
    p = np.empty((X.shape[0], 1))
    h_x = np.empty((X.shape[0], 3))
    for q in range(X.shape[0]): # for each testing sample
        a_1 = np.vstack((1, X[q].reshape(-1, 1)))
        z_2 = np.dot(Theta1, a_1)
        a_2 = np.vstack((1, sigmoid(z_2)))
        z_3 = np.dot(Theta2, a_2)
        h_x[q] = sigmoid(z_3).flatten()
        p[q] = np.argmax(z_3) + 1

    return p, h_x


if __name__ == "__main__":
    data = loadmat('input/HW7_Data2_full.mat') 
    X = data['X']
    weights = loadmat('input/HW7_weights_3_full.mat')
    Theta1 = weights['Theta1']
    Theta2 = weights['Theta2']
    p, h_x = predict(Theta1, Theta2, X)
    print(p)