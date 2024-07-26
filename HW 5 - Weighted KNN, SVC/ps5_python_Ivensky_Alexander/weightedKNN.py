import numpy as np 
from scipy.spatial.distance import cdist
from scipy.io import loadmat

# X_train: m x (n + 1) - features matrix
# y_train: m x 1 - labels vector
# X_test: d x (n + 1) - testing feature matrix
# sigma: scalar - bandwidth of Gaussian weighting function
# y_predict: d x 1 - predicted labels for test instances
def weightedKNN(X_train, y_train, X_test, sigma):
    d = X_test.shape[0]
    dist = cdist(X_test, X_train, metric='euclidean')
    W = np.exp((-(dist ** 2)) / (sigma ** 2))
    C = np.unique(y_train) 
    weighted_vote = np.empty((d, C.shape[0]))
    y_predict = np.empty((d, 1))
    for i in range(d):
        for j, c in enumerate(C): 
            weighted_vote[i, j] = np.sum(W[i, np.where(y_train == c)])
        y_predict[i] = C[np.argmax(weighted_vote[i, :])]
    return y_predict

# ----- ignore this, actual answer to problem is in ps5.py
if __name__ == "__main__":
    data = loadmat('input/hw4_data3.mat')
    X_test = data["X_test"]
    X_train = data["X_train"]
    y_test = data["y_test"]
    y_train = data["y_train"]
    sigma = np.array([0.01, 0.07, 0.15, 1.5, 3, 4.50])
    # y_pred = np.empty((y_test.shape[0], sigma.shape[0]))

    # for i, s in enumerate(sigma):
    #     y_pred[:, i:i+1] = weightedKNN(X_train, y_train, X_test, s)
    y_pred = weightedKNN(X_train, y_train, X_test, 0.07)
    print(y_pred)
    