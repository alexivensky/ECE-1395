# Alex Ivensky
# ECE 1395
# Homework 4

import numpy as np
import matplotlib.pyplot as plt
import scipy
from normalEqn import *
from computeCost import *
from logReg_multi import *
from sklearn.neighbors import *
from sklearn.metrics import *

# 1b

data1 = scipy.io.loadmat('input/hw4_data1.mat')
X_nonbias = data1["X_data"]
y = data1["y"]
X = np.hstack((np.ones((X_nonbias.shape[0], 1)), X_nonbias))
print("(1b) Feature matrix size: ", X.shape)

# 1c

lamb = np.array([0, 0.001, 0.003, 0.005, 0.007, 0.009, 0.012, 0.017])
iters = 20
lamb_num = 8
cutoff = int(0.85 * X.shape[0])
training_error = np.zeros((iters, lamb_num))
testing_error = np.zeros((iters, lamb_num))

for i in range(iters):
    combined = np.hstack((X, y))
    np.random.shuffle(combined)
    X_train = combined[:cutoff, :-1]
    X_test = combined[cutoff:, :-1]
    y_train = combined[:cutoff, -1].reshape(-1, 1)
    y_test = combined[cutoff:, -1].reshape(-1, 1)
    for j, l in enumerate(lamb):
        train_theta = Reg_normalEqn(X_train, y_train, l)
        test_theta = Reg_normalEqn(X_test, y_test, l)
        training_error[i, j] = computeCost(X_train, y_train, train_theta)
        testing_error[i, j] = computeCost(X_test, y_test, test_theta)

avg_train_error = np.mean(training_error, axis=0)
avg_test_error = np.mean(testing_error, axis=0)
plt.xlabel("λ")
plt.ylabel("Average Error")
plt.plot(lamb, avg_train_error, color='red', marker='*', label="Training Error")
plt.plot(lamb, avg_test_error, color='blue', marker='o', label="Testing Error")
plt.legend()
plt.savefig('output/ps4-1-a.png')
plt.close()
print(avg_train_error)
print(avg_test_error)

# 2

# 2a

data2 = scipy.io.loadmat('input/hw4_data2.mat')
X1 = data2["X1"]
X2 = data2["X2"]
X3 = data2["X3"]
X4 = data2["X4"]
X5 = data2["X5"]
y1 = data2["y1"]
y2 = data2["y2"]
y3 = data2["y3"]
y4 = data2["y4"]
y5 = data2["y5"]
K = np.arange(1, 15, 2)

# first classifier data
X_train1 = np.vstack((X1, X2, X3, X4))
y_train1 = np.vstack((y1, y2, y3, y4))
X_test1 = X5
y_test1 = y5

acc1 = np.empty(K.shape)

for i, k in enumerate(K):
    c = KNeighborsClassifier(n_neighbors=k)
    c.fit(X_train1, y_train1.flatten())
    y_pred = c.predict(X_test1)
    acc1[i] = accuracy_score(y_test1.flatten(), y_pred.flatten())
    
# second classifier data
X_train2 = np.vstack((X1, X2, X3, X5))
y_train2 = np.vstack((y1, y2, y3, y5))
X_test2 = X4
y_test2 = y4

acc2 = np.empty(K.shape)

for i, k in enumerate(K):
    c = KNeighborsClassifier(n_neighbors=k)
    c.fit(X_train2, y_train2.flatten())
    y_pred = c.predict(X_test2)
    acc2[i] = accuracy_score(y_test2.flatten(), y_pred.flatten())
    
# third classifier data
X_train3 = np.vstack((X1, X2, X4, X5))
y_train3 = np.vstack((y1, y2, y4, y5))
X_test3 = X3
y_test3 = y3

acc3 = np.empty(K.shape)

for i, k in enumerate(K):
    c = KNeighborsClassifier(n_neighbors=k)
    c.fit(X_train3, y_train3.flatten())
    y_pred = c.predict(X_test3)
    acc3[i] = accuracy_score(y_test3.flatten(), y_pred.flatten())
    
# fourth classifier data
X_train4 = np.vstack((X1, X3, X4, X5))
y_train4 = np.vstack((y1, y3, y4, y5))
X_test4 = X2
y_test4 = y2

acc4 = np.empty(K.shape)

for i, k in enumerate(K):
    c = KNeighborsClassifier(n_neighbors=k)
    c.fit(X_train4, y_train4.flatten())
    y_pred = c.predict(X_test4)
    acc4[i] = accuracy_score(y_test4.flatten(), y_pred.flatten())
    
# fifth classifier data
X_train5 = np.vstack((X2, X3, X4, X5))
y_train5 = np.vstack((y2, y3, y4, y5))
X_test5 = X1
y_test5 = y1

acc5 = np.empty(K.shape)

for i, k in enumerate(K):
    c = KNeighborsClassifier(n_neighbors=k)
    c.fit(X_train5, y_train5.flatten())
    y_pred = c.predict(X_test5)
    acc5[i] = accuracy_score(y_test5.flatten(), y_pred.flatten())
    

acc_mat = np.vstack((acc1, acc2, acc3, acc4, acc5))
acc = np.mean(acc_mat, axis=0)
plt.xlabel("K")
plt.ylabel("Average Accuracy")
plt.plot(K, acc, marker='o')
plt.savefig('output/ps4-2-a.png')
plt.close()

# 3

data3 = scipy.io.loadmat('input/hw4_data3.mat')

print(data3.keys())
X_train = data3["X_train"]
X_test = data3["X_test"]
y_train = data3["y_train"]
y_test = data3["y_test"]
y_pred = logReg_multi(X_train, y_train, X_test)
train = logReg_multi(X_train, y_train, X_train)
print("Testing Accuracy: ", accuracy_score(y_test, y_pred))
print("Training Accuracy: ", accuracy_score(y_train, train))