# Alex Ivensky
# ECE 1395
# HW 6

import numpy as np
import pandas
import math

# Data Description and Preprocessing

iris_data = pandas.read_csv("input/iris_dataset.csv", header=None)
X = iris_data.iloc[:, :4].values
y = iris_data.iloc[:, -1].values.reshape(-1,1)

combined = np.hstack((X, y))
np.random.shuffle(combined)

X_train = combined[0:125, :4]
X_test = combined[125:150, :4]
y_train = combined[0:125, -1].reshape(-1,1)
y_test = combined[125:150, -1].reshape(-1,1)

X_train_1 = X_train[np.where(y_train == 1)[0]]
X_train_2 = X_train[np.where(y_train == 2)[0]]
X_train_3 = X_train[np.where(y_train == 3)[0]]

print("Sizes of X_trains: ")
print("X_train_1 : ", X_train_1.shape)
print("X_train_2 : ", X_train_2.shape)
print("X_train_3 : ", X_train_3.shape)

# 1. Naive Bayes Classifier

X_train_1_means = np.mean(X_train_1, axis=0).reshape(-1, 1) # mu_1|1, mu_2|1, mu_3|1, mu_4|1
X_train_2_means = np.mean(X_train_2, axis=0).reshape(-1, 1)
X_train_3_means = np.mean(X_train_3, axis=0).reshape(-1, 1)
means = np.hstack((X_train_1_means, X_train_2_means, X_train_3_means))

X_train_1_std = np.std(X_train_1, axis=0).reshape(-1, 1)
X_train_2_std = np.std(X_train_2, axis=0).reshape(-1, 1)
X_train_3_std = np.std(X_train_3, axis=0).reshape(-1, 1)
stds = np.hstack((X_train_1_std, X_train_2_std, X_train_3_std)) # columns are labels, rows are features

print("X_train Mean Values:")
print(means)

print("X_train Standard Deviations:")
print(stds)

pxj_w1 = np.empty(X_test.shape[1])
pxj_w2 = np.empty(X_test.shape[1])
pxj_w3 = np.empty(X_test.shape[1])
acc = 0
for i in range(X_test.shape[0]): # for each testing sample
    for j in range(X_test.shape[1]): # for each feature
        pxj_w1[j] = (1/(math.sqrt(2*math.pi)*stds[j, 0])) * np.exp(-(X_test[i, j] - means[j, 0])**2/(2*(stds[j, 0]**2)))
        pxj_w2[j] = (1/(math.sqrt(2*math.pi)*stds[j, 1])) * np.exp(-(X_test[i, j] - means[j, 1])**2/(2*(stds[j, 1]**2)))
        pxj_w3[j] = (1/(math.sqrt(2*math.pi)*stds[j, 2])) * np.exp(-(X_test[i, j] - means[j, 2])**2/(2*(stds[j, 2]**2)))

    ln_pxj_w1 = np.sum(np.log(pxj_w1))
    ln_pxj_w2 = np.sum(np.log(pxj_w2))
    ln_pxj_w3 = np.sum(np.log(pxj_w3))

    ln_pw1_x = ln_pxj_w1 + np.log(1/3)
    ln_pw2_x = ln_pxj_w2 + np.log(1/3)
    ln_pw3_x = ln_pxj_w3 + np.log(1/3)
    
    label = np.argmax([ln_pw1_x, ln_pw2_x, ln_pw3_x]) + 1
    if label == y_test[i, 0]:
        acc += 1

print("Accuracy: ", (acc / y_test.shape[0]))

# Max Likelihood and Discriminant Function for Classification

Sigma_1 = np.cov(X_train_1.T)
Sigma_2 = np.cov(X_train_2.T)
Sigma_3 = np.cov(X_train_3.T)
print("S1 size: ", Sigma_1.shape)
print(Sigma_1)
print("S2 size: ", Sigma_2.shape)
print(Sigma_2)
print("S3 size: ", Sigma_3.shape)
print(Sigma_3)

mean1 = X_train_1_means
mean2 = X_train_2_means
mean3 = X_train_3_means

print("Mean 1 size: ", mean1.shape)
print(mean1)
print("Mean 2 size: ", mean2.shape)
print(mean2)
print("Mean 3 size: ", mean3.shape)
print(mean3)

g1_x = np.empty((X_test.shape[0], 1))
g2_x = np.empty((X_test.shape[0], 1))
g3_x = np.empty((X_test.shape[0], 1))

MLFacc = 0

for i in range(X_test.shape[0]):
    g1_x[i, 0] = (-1/2)*(X_test[i, :].reshape(-1,1) - mean1).T @ np.linalg.inv(Sigma_1) @ (X_test[i, :].reshape(-1,1) - mean1) + np.log(1/3) - (mean1.shape[0]/2)*np.log(2*math.pi) - (1/2)*np.log(np.linalg.det(Sigma_1))
    g2_x[i, 0] = (-1/2)*(X_test[i, :].reshape(-1,1) - mean2).T @ np.linalg.inv(Sigma_2) @ (X_test[i, :].reshape(-1,1) - mean2) + np.log(1/3) - (mean2.shape[0]/2)*np.log(2*math.pi) - (1/2)*np.log(np.linalg.det(Sigma_2))
    g3_x[i, 0] = (-1/2)*(X_test[i, :].reshape(-1,1) - mean3).T @ np.linalg.inv(Sigma_3) @ (X_test[i, :].reshape(-1,1) - mean3) + np.log(1/3) - (mean3.shape[0]/2)*np.log(2*math.pi) - (1/2)*np.log(np.linalg.det(Sigma_3))
    label = np.argmax([g1_x[i,0], g2_x[i,0], g3_x[i,0]]) + 1
    if label == y_test[i, 0]:
        MLFacc += 1

print("MLF accuracy: ", MLFacc / y_test.shape[0])