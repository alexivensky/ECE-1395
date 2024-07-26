import numpy as np
from scipy.spatial.distance import cdist

# X: (m, n) 
    # m : number of samples
    # n : dimensionality of each feature vector
    # K : number of clusters
    # iters : number of iterations to run for K-means
# returns
    # ids : (m, 1) vector containing data membership IDs of each sample 
    # means : (K, n) matrix containing the centers/means of each cluster
    # ssd : scalar giving final SSD of the clustering (sum of squared distances)
def kmeans_single(X, K, iters):
    m, n = X.shape
    ids = np.empty((m, 1))
    means = np.empty((K, n))
    min_X = np.min(X, axis=0)
    max_X = np.max(X, axis=0)
    for i in range(K):
        means[i, :] = min_X + np.random.rand(n) * (max_X - min_X)
    for _ in range(iters):
        distances = cdist(X, means, 'euclidean')
        ids = np.argmin(distances, axis=1)
        ssd = 0
        for i in range(K):
            members = (ids == i)
            if np.sum(members):
                means[i, :] = np.mean(X[members, :], axis=0)
                ssd += np.sum(distances[members, i] ** 2)
    return ids, means, ssd


if __name__ == "__main__":
    m = 100
    n = 2
    X = np.random.rand(m, n)
    ids, means, ssd = kmeans_single(X, 3, 10)
    