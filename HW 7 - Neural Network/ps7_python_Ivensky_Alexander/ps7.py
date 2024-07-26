# Alex Ivensky
# ECE 1395
# Homework 7

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from predict import predict
from sklearn.metrics import accuracy_score
from nnCost import nnCost
from sigmoidGradient import sigmoidGradient
from sGD import sGD
import time

# thought it'd be fun to time the program
# it takes 28 minutes on my computer
start = time.time()

# Data Description and Preprocessing

data = loadmat('input/HW7_Data2_full.mat') # importing .mat file
X = data['X']
y_labels = data['y_labels']
text_labels = { # text versions for each label
    1: "Airplane",
    2: "Automobile",
    3: "Truck"
}
# plotting 16 random images on subplot
rand_pics = np.random.randint(0, 15000, size=16)
fig, ax = plt.subplots(4, 4, figsize=(16, 16))
for i in range(4):
    for j in range(4):
        idx = 4*i + j
        img = X[rand_pics[idx], :].reshape(32, 32).T
        ax[i, j].set_title(text_labels[y_labels[rand_pics[idx], 0]])
        ax[i, j].imshow(img, cmap='gray')
plt.savefig('output/ps7-0-a-1.png')

# test-train split
X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=(2/15))

# Forward Propagation

weights = loadmat('input/HW7_weights_3_full.mat')
Theta1 = weights['Theta1']
Theta2 = weights['Theta2']
p, h_x = predict(Theta1, Theta2, X)
acc = accuracy_score(y_labels, p)
print("1b. Accuracy: ", acc)

# Cost Function

lda = [0.1, 1, 2]
for i in range(3):
    print(f"Cost for Lambda = {lda[i]}: {nnCost(Theta1, Theta2, X, y_labels, 3, lda[i])}")

# Sigmoid Gradient

z = np.array([-10, 0, 10]).reshape(-1, 1)
print("Sigmoid Gradient Test: \n", sigmoidGradient(z))

# Testing the Network

### MaxEpochs = 50
# l = 0.1

Theta1, Theta2 = sGD(1024, 40, 3, X_train, y_train, 0.1, 0.0001, 50)
p, h_x = predict(Theta1, Theta2, X_train)
print("Training Accuracy for 50/0.1: ", accuracy_score(y_train, p))
cost = nnCost(Theta1, Theta2, X_train, y_train, 3, 0.0001)
print("Training Cost: ", cost)
p, h_x = predict(Theta1, Theta2, X_test)
print("Testing Accuracy for 50/0.1: ", accuracy_score(y_test, p))
cost = nnCost(Theta1, Theta2, X_test, y_test, 3, 0.0001)
print("Testing Cost: ", cost)

# l = 1

Theta1, Theta2 = sGD(1024, 40, 3, X_train, y_train, 1, 0.0001, 50)
p, h_x = predict(Theta1, Theta2, X_train)
print("Training Accuracy for 50/1: ", accuracy_score(y_train, p))
cost = nnCost(Theta1, Theta2, X_train, y_train, 3, 0.0001)
print("Training Cost: ", cost)
p, h_x = predict(Theta1, Theta2, X_test)
print("Testing Accuracy for 50/1: ", accuracy_score(y_test, p))
cost = nnCost(Theta1, Theta2, X_test, y_test, 3, 0.0001)
print("Testing Cost: ", cost)

# l = 2

Theta1, Theta2 = sGD(1024, 40, 3, X_train, y_train, 2, 0.0001, 50)
p, h_x = predict(Theta1, Theta2, X_train)
print("Training Accuracy for 50/2: ", accuracy_score(y_train, p))
cost = nnCost(Theta1, Theta2, X_train, y_train, 3, 0.0001)
print("Training Cost: ", cost)
p, h_x = predict(Theta1, Theta2, X_test)
print("Testing Accuracy for 50/2: ", accuracy_score(y_test, p))
cost = nnCost(Theta1, Theta2, X_test, y_test, 3, 0.0001)
print("Testing Cost: ", cost)

### MaxEpochs = 300
# l = 0.1

Theta1, Theta2 = sGD(1024, 40, 3, X_train, y_train, 0.1, 0.0001, 300)
p, h_x = predict(Theta1, Theta2, X_train)
print("Training Accuracy for 300/0.1: ", accuracy_score(y_train, p))
cost = nnCost(Theta1, Theta2, X_train, y_train, 3, 0.0001)
print("Training Cost: ", cost)
p, h_x = predict(Theta1, Theta2, X_test)
print("Testing Accuracy for 300/0.1: ", accuracy_score(y_test, p))
cost = nnCost(Theta1, Theta2, X_test, y_test, 3, 0.0001)
print("Testing Cost: ", cost)

# l = 1

Theta1, Theta2 = sGD(1024, 40, 3, X_train, y_train, 1, 0.0001, 300)
p, h_x = predict(Theta1, Theta2, X_train)
print("Training Accuracy for 300/1: ", accuracy_score(y_train, p))
cost = nnCost(Theta1, Theta2, X_train, y_train, 3, 0.0001)
print("Training Cost: ", cost)
p, h_x = predict(Theta1, Theta2, X_test)
print("Testing Accuracy for 300/1: ", accuracy_score(y_test, p))
cost = nnCost(Theta1, Theta2, X_test, y_test, 3, 0.0001)
print("Testing Cost: ", cost)

# l = 2

Theta1, Theta2 = sGD(1024, 40, 3, X_train, y_train, 2, 0.0001, 300)
p, h_x = predict(Theta1, Theta2, X_train)
print("Training Accuracy for 300/2: ", accuracy_score(y_train, p))
cost = nnCost(Theta1, Theta2, X_train, y_train, 3, 0.0001)
print("Training Cost: ", cost)
p, h_x = predict(Theta1, Theta2, X_test)
print("Testing Accuracy for 300/2: ", accuracy_score(y_test, p))
cost = nnCost(Theta1, Theta2, X_test, y_test, 3, 0.0001)
print("Testing Cost: ", cost)

print(f"\nTOTAL TIME: {(time.time() - start)/60} minutes.")