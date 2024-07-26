# Alex Ivensky
# ECE 1395
# Homework 3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sigmoid import *
import scipy
from costFunction import *
from gradFunction import *

# 1a

data1 = np.loadtxt("input/hw3_data1.txt", delimiter=",")
x0 = np.ones((data1.shape[0], 1))
X = np.hstack((x0, data1[:, [0,1]]))
y = data1[:, 2].reshape(-1, 1)
print("1a.")
print("Size of X: ", X.shape)
print("Size of y: ", y.shape)

# 1b

plt.title("HW3 #1b")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
for i in range(len(y)):
    marker = '*' if y[i] == 1 else 'x'
    color = 'green' if y[i] == 1 else 'red'
    plt.scatter(X[i, 1], X[i, 2], c=color, marker=marker)
plt.savefig("output/ps3-1-b.png")
plt.close()

# 1c

cutoff = int(0.9*X.shape[0])
comp = X.shape[0] - cutoff
combined = np.hstack((X, y))
np.random.shuffle(combined)
X_train = combined[0:cutoff, 0:3]
X_test = combined[cutoff:X.shape[0], 0:3]
y_train = combined[0:cutoff, 3].reshape(-1, 1)
y_test = combined[cutoff:X.shape[0], 3].reshape(-1, 1)

# 1d

z = np.arange(-15, 15, 0.01)
gz = np.array([sigmoid(i) for i in z])
plt.plot(z, gz)
plt.savefig("output/ps3-1-c.png")
plt.close()

# 1f

theta_init = np.zeros((X_train.shape[1]))
print(theta_init.shape) # 3 x 1
print(X_train.shape) # 90 x 3
print(y_train.shape) # 90 x 1
theta = scipy.optimize.fmin_bfgs(costFunction, theta_init, fprime=gradFunction, args=(X_train, y_train))
print("Optimized theta:\n",theta)
print("Optimized cost:", costFunction(theta, X_train, y_train))

# 1g

plt.title("HW3 #1g")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
for i in range(len(y)):
    marker = '*' if y[i] == 1 else 'x'
    color = 'green' if y[i] == 1 else 'red'
    plt.scatter(X[i, 1], X[i, 2], c=color, marker=marker)
minX = np.min(X_train[:, 1])
maxX = np.max(X_train[:, 1])
decision_boundary_x = np.array([minX, maxX])
decision_boundary_y = -(theta[0] + theta[1] * decision_boundary_x) / theta[2]
plt.plot(decision_boundary_x, decision_boundary_y, '--', color='blue')
plt.savefig("output/ps3-1-f.png")
plt.close()

# 1h

h = sigmoid(X_train @ theta)
predictions = (h >= 0.5).astype(int)
correct = np.sum(predictions == y_train.flatten())
accuracy = correct / y_train.shape[0]
print("Accuracy: ", accuracy)

# 1i

test1 = 60
test2 = 65
test_scores = np.array([1, test1, test2])
z = sigmoid(theta @ test_scores)
print("Probability of admission with test1 = 60 and test2 = 65: ", z)

# 2a

path2 = 'input/hw3_data2.csv'
data2 = pd.read_csv(path2, header=None)
x1 = data2.iloc[:,0].values.reshape(-1,1)
y = data2.iloc[:,1].values.reshape(-1,1)
x0 = np.ones((x1.shape[0], 1))
x2 = x1 * x1
X = np.hstack((x0, x1, x2))
theta2 = np.linalg.pinv(X.T @ X) @ (X.T @ y)
print("2a theta:\n", theta2)

# 2b

x_values = np.linspace(min(x1), max(x1), 100).reshape(-1, 1)
x0_plot = np.ones((x_values.shape[0], 1))
x2_plot = x_values * x_values
X_plot = np.hstack((x0_plot, x_values, x2_plot))
y_plot = X_plot @ theta2
plt.scatter(x1, y, color='red')
plt.plot(x_values, y_plot, color='blue')
plt.xlabel('Population in thousands, n')
plt.ylabel('Profit')
plt.title('HW3, 2b')
plt.savefig("output/hw3-2-b.png")








