import numpy as np
from kmeans_single import kmeans_single


def kmeans_multiple(X, K, iters, R):
    ids_list = []
    means_list = []
    ssds_list = []
    for _ in range(R):
        ids, means, ssd = kmeans_single(X, K, iters)
        ids_list.append(ids)
        means_list.append(means)
        ssds_list.append(ssd)
    idx = np.argmin(ssds_list)
    return ids_list[idx], means_list[idx], ssds_list[idx]


if __name__ == "__main__":
    m = 100
    n = 2
    X = np.random.rand(m, n)
    ids, means, ssd = kmeans_multiple(X, 3, 10, 10)
    print(ssd)
