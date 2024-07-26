# Alex Ivensky
# ECE 1395
# Homework 5

import numpy as np 
import shutil
import os
import glob
import cv2
import matplotlib.pyplot as plt
from weightedKNN import weightedKNN
from scipy.io import loadmat
from scipy.linalg import eig
from scipy.sparse.linalg import eigs
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import time

# 1b
data = loadmat('input/hw4_data3.mat')
X_test = data["X_test"]
X_train = data["X_train"]
y_test = data["y_test"]
y_train = data["y_train"]
sigma = np.array([0.01, 0.07, 0.15, 1.5, 3, 4.50])
print("Problem 1:")
for i, s in enumerate(sigma):
    y_pred = weightedKNN(X_train, y_train, X_test, s)
    print(f"Sigma: {s}, Accuracy: {accuracy_score(y_test, y_pred)}")
    
# 2
print("Problem 2:")
#### choosing photos
# deleting all files in train and test
files = glob.glob('input/train/*')
for f in files:
    os.remove(f)
files = glob.glob('input/test/*')
for f in files:
    os.remove(f)
# randomly choosing new testing and training sets
for i in range(1, 41): 
    phtArr = np.arange(1, 11, 1)
    np.random.shuffle(phtArr)
    for j in range(10):
        if j < 8:
            shutil.copy(f'input/all/s{i}-{phtArr[j]}.pgm', f'input/train/')
        else:
            shutil.copy(f'input/all/s{i}-{phtArr[j]}.pgm', f'input/test/')
# choosing three random images from training set
files = glob.glob('input/train/*')
np.random.shuffle(files)
fig, ax = plt.subplots(1, 3)
for i in range(3):
    img = plt.imread(files[i])
    ax[i].imshow(img, cmap='gray')
    ax[i].axis('off')
    ax[i].set_title(files[i].split('/')[-1])
plt.savefig('output/ps5-2-0.png')
plt.close()

# 2.1 a

T = np.hstack([cv2.imread(file, cv2.IMREAD_UNCHANGED).flatten().reshape(-1, 1) for file in glob.glob('input/train/*')]) # wow
plt.imshow(T, cmap='gray')
plt.savefig('output/ps5-1-a.png')
plt.close()

# 2.1 b

T_mean = np.mean(T, axis=1)
img_mean = np.reshape(T_mean, (112, 92))
plt.imshow(img_mean, cmap='gray')
plt.savefig('output/ps5-2-1-b.png')
plt.close()

# 2.1 c

m = np.tile(T_mean[:, np.newaxis], 320)
A = T - m
C = A @ A.T
plt.imshow(C, cmap='gray')
plt.savefig('output/ps5-2-1-c.png')
plt.close()

# 2.1 d

A_eigval, A_eigvec = eig(A.T @ A)
np.sort(A_eigval)
sum_eig = np.sum(A_eigval)
k = np.array([])
v_k = np.array([])
for i in range(A_eigval.shape[0]):
    v_k = np.append(v_k, (A_eigval[i] / sum_eig))
    k = np.append(k, i)
    if np.sum(v_k) >= 0.95:
        break
plt.plot(k, v_k)
plt.savefig('output/ps5-2-1-d.png')
print("k: ", np.argmax(k))

# 2.1 e

vals, vecs = eigs(C, k=np.argmax(k))
U = np.real(vecs)
fig, ax = plt.subplots(3, 3)
for i in range(3):
    for j in range(3):
        idx = i*3 + j
        eigenface = U[:, idx].reshape((112, 92)) 
        ax[i, j].imshow(eigenface, cmap='gray')
plt.savefig('output/ps5-2-1-e.png')
plt.close()
print("U Dim: ", U.shape)

# 2.2 a

w = []  
labels = []
for file in glob.glob('input/train/*'):
    I = cv2.imread(file, cv2.IMREAD_UNCHANGED).flatten()
    w_i = U.T @ (I - T_mean)
    name = os.path.basename(file)
    labels.append(name[name.find('s')+1:name.find('-')])
    w.append(w_i)  
W_training = np.array(w)  
labels = np.array(labels).reshape(-1,1)

# 2.2 b

w = []  
true_class = []
for file in glob.glob('input/test/*'):
    I = cv2.imread(file, cv2.IMREAD_UNCHANGED).flatten()
    w_i = U.T @ (I - T_mean)
    name = os.path.basename(file)
    true_class.append(name[name.find('s')+1:name.find('-')])
    w.append(w_i)  
W_testing = np.array(w)  
true_class = np.array(true_class).reshape(-1,1)

print("W_training Dim: ", W_training.shape)
print("W_testing Dim: ", W_testing.shape)


# 2.3 a

k_vals = [1, 3, 4, 7, 9, 11]
for i in k_vals:
    n = KNeighborsClassifier(n_neighbors=i)
    n.fit(W_training, labels.ravel())
    labels_pred = n.predict(W_testing)
    print(f"k: {i}, Accuracy: {accuracy_score(true_class, labels_pred)}")
    
# 2.3 b

print("\n")

# Linear OVO

lin_ovo = SVC(kernel='linear', decision_function_shape='ovo')
start = time.time()
lin_ovo.fit(W_training, labels.ravel())
print("Linear OVO Training Time: ", time.time() - start)
start = time.time()
lin_ovo_pred = lin_ovo.predict(W_testing)
print("Linear OVO Testing Time: ", time.time() - start)
print("Linear OVO Testing Accuracy: ", accuracy_score(true_class, lin_ovo_pred))
print("\n")

# Linear OVR

lin_ovr = SVC(kernel='linear', decision_function_shape='ovr')
start = time.time()
lin_ovr.fit(W_training, labels.ravel())
print("Linear OVR Training Time: ", time.time() - start)
start = time.time()
lin_ovr_pred = lin_ovr.predict(W_testing)
print("Linear OVR Testing Time: ", time.time() - start)
print("Linear OVR Testing Accuracy: ", accuracy_score(true_class, lin_ovr_pred))
print("\n")

# Polynomial OVO

poly_ovo = SVC(kernel='poly', decision_function_shape='ovo')
start = time.time()
poly_ovo.fit(W_training, labels.ravel())
print("Poly OVO Training Time: ", time.time() - start)
start = time.time()
poly_ovo_pred = poly_ovo.predict(W_testing)
print("Poly OVO Testing Time: ", time.time() - start)
print("Poly OVO Testing Accuracy: ", accuracy_score(true_class, poly_ovo_pred))
print("\n")

# Polynomial OVR

poly_ovr = SVC(kernel='poly', decision_function_shape='ovr')
start = time.time()
poly_ovr.fit(W_training, labels.ravel())
print("Poly OVR Training Time: ", time.time() - start)
start = time.time()
poly_ovr_pred = poly_ovr.predict(W_testing)
print("Poly OVR Testing Time: ", time.time() - start)
print("Poly OVR Testing Accuracy: ", accuracy_score(true_class, poly_ovr_pred))
print("\n")

# RBF OVO

rbf_ovo = SVC(kernel='rbf', decision_function_shape='ovo')
start = time.time()
rbf_ovo.fit(W_training, labels.ravel())
print("RBF OVO Training Time: ", time.time() - start)
start = time.time()
rbf_ovo_pred = rbf_ovo.predict(W_testing)
print("RBF OVO Testing Time: ", time.time() - start)
print("RBF OVO Testing Accuracy: ", accuracy_score(true_class, rbf_ovo_pred))
print("\n")

# RBF OVR

rbf_ovr = SVC(kernel='rbf', decision_function_shape='ovr')
start = time.time()
rbf_ovr.fit(W_training, labels.ravel())
print("RBF OVR Training Time: ", time.time() - start)
start = time.time()
rbf_ovr_pred = rbf_ovr.predict(W_testing)
print("RBF OVR Testing Time: ", time.time() - start)
print("RBF OVR Testing Accuracy: ", accuracy_score(true_class, rbf_ovr_pred))
