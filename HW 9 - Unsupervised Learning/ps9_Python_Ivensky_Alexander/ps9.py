# Alex Ivensky
# ECE 1395
# Homework 9

import cv2
import numpy as np
from Segment_kmeans import *
import matplotlib.pyplot as plt

# performing clustering

images = ['input/HW9_S_images/im1.jpg', 'input/HW9_S_images/im2.jpg', 'input/HW9_S_images/im3.png']
K = [3, 5, 7]
iters = [7, 13, 20]
R = [5, 15, 30]

for img in images:
    for k in K:
        for it in iters:
            for r in R:
                im_out = Segment_kmeans(img, k, it, r)
                plt.imshow(im_out)
                plt.title(f"Img = {img[19:22]}, K = {k}, Iterations = {it}, R = {r}")
                plt.savefig(f'output/ps9-{img[19:22]}-{k}-{it}-{r}.png')



                