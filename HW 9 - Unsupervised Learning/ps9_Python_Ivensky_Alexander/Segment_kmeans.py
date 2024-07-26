import cv2
import numpy as np
from kmeans_multiple import *
import matplotlib.pyplot as plt

def Segment_kmeans(im_in, K, iters, R):
    im_in = cv2.imread(im_in)
    im = im_in.astype(np.float64)
    im = cv2.resize(im, (100, 100))
    H, W, _ = im.shape
    X = np.reshape(im, (H*W, 3))
    ids, means, _ = kmeans_multiple(X, K, iters, R)
    recolored_im = np.zeros_like(X)
    for i in range(K):
        members = (ids == i)
        recolored_im[members, :] = means[i]
    recolored_im = np.reshape(recolored_im, (H, W, 3))
    im_out = recolored_im.astype(np.uint8)
    return im_out
    
if __name__ == "__main__":
    m = 100
    n = 2
    im_in = "input/HW9_S_images/im3.png"
    im_out = Segment_kmeans(im_in, 3, 200, 30)
    plt.imshow(im_out)
    plt.show()