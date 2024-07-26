# Alex Ivensky
# ECE 1395
# HW 2

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gradientDescent import *
from normalEqn import *

# 4a

path1 = 'input/hw2_data1.csv'
data1 = pd.read_csv(path1, header=None)
x1 = data1.iloc[:,0].values.reshape(-1,1)
y = data1.iloc[:,1].values.reshape(-1,1)

# 4b

plt.title("HW2 #4b")
plt.xlabel("Horse power of a car in 100s")
plt.ylabel("Price in $1000s")
plt.scatter(x1, y)
plt.savefig('output/ps2-4-b.png')
plt.close()

# 4c

x0 = np.ones((x1.shape[0], 1))
X = np.hstack((x0, x1))
print("4c Dimensions:")
print(f"X: {X.shape}")
print(f"y: {y.shape}")

# 4d

combined = np.hstack((X, y))
np.random.shuffle(combined)

X_train = combined[0:161, 0:2]
X_test = combined[161:179, 0:2]

y_train = combined[0:161, 2:3]
y_test = combined[161:179, 2:3]

# 4e

theta_gd, cost_gd = gradientDescent(X_train, y_train, 0.3, 500)
iter = np.arange(1, 501)
print("4e theta:\n", theta_gd)

plt.title("HW2 #4e")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.scatter(iter, cost_gd)
plt.savefig('output/ps2-4-e.png')
plt.close()

# 4f

minHP = np.min(X)
maxHP = np.max(X)
minPoint = theta_gd[0, 0] + theta_gd[1, 0] * minHP
maxPoint = theta_gd[0, 0] + theta_gd[1, 0] * maxHP
h = np.array([minPoint, maxPoint])
x_h = np.array([minHP, maxHP])
plt.title("HW2 #4f")
plt.xlabel("Horse power of a car in 100s")
plt.ylabel("Price in $1000s")
plt.scatter(x1, y)
plt.plot(x_h, h, '--', color='red')
plt.savefig('output/ps2-4-f.png')
plt.close()

# 4g

y_pred = theta_gd[0, 0] + theta_gd[1, 0] * X_test
y_pred_cost = computeCost(y_pred, y_test, theta_gd)
print("4g Error\n", y_pred_cost)

# 4h

theta_ne = normalEqn(X_test, y_test)
y_ne = theta_ne[0, 0] + theta_ne[1, 0] * X_test
y_ne_cost = computeCost(y_ne, y_test, theta_ne)
print("4h Error\n", y_ne_cost)

alpha = np.array([0.001, 0.003, 0.03, 3])
iter_a = np.arange(1, 301)

theta_1, cost_1 = gradientDescent(X, y, alpha[0], 300)

plt.title("HW2 #4i, 1")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.scatter(iter_a, cost_1)
plt.savefig('output/ps2-4-i-1.png')
plt.close()

theta_2, cost_2 = gradientDescent(X, y, alpha[1], 300)

plt.title("HW2 #4i, 2")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.scatter(iter_a, cost_2)
plt.savefig('output/ps2-4-i-2.png')
plt.close()

theta_3, cost_3 = gradientDescent(X, y, alpha[2], 300)

plt.title("HW2 #4i, 3")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.scatter(iter_a, cost_3)
plt.savefig('output/ps2-4-i-3.png')
plt.close()

theta_4, cost_4 = gradientDescent(X, y, alpha[3], 300)

plt.title("HW2 #4i, 4")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.scatter(iter_a, cost_4)
plt.savefig('output/ps2-4-i-4.png')
plt.close()

# 5a

path2 = 'input/hw2_data3.csv'
data2 = pd.read_csv(path2, header=None)
x1 = data2.iloc[:,0].values.reshape(-1,1)
x2 = data2.iloc[:,1].values.reshape(-1,1)
y = data2.iloc[:,2].values.reshape(-1,1)

x1_mean = np.mean(x1)
x2_mean = np.mean(x2)
y_mean = np.mean(y)

x1_std = np.std(x1)
x2_std = np.std(x2)
y_std = np.std(y)

print(f"X1 mean: {x1_mean}")
print(f"X2 mean: {x2_mean}")
print(f"y mean: {y_mean}")
print(f"X1 stddev: {x1_std}")
print(f"X2 stddev: {x2_std}")
print(f"y stddev: {y_std}")

x1 = (x1 - x1_mean) / x1_std
x2 = (x2 - x2_mean) / x2_std
y = (y - y_mean) / y_std

x0 = np.ones(x1.shape)

X = np.hstack((x0, x1, x2))

theta_5, cost_5 = gradientDescent(X, y, 0.01, 750)
print("X dim", theta_5.shape)
print("y dim", y.shape)
print("Theta: \n", theta_5)

iter_5 = np.arange(1, 751)
plt.title("HW2, #5b")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.scatter(iter_5, cost_5)
plt.savefig('output/ps2-5-b.png')
plt.close()

eng_size = (2300 - x1_mean) / x1_std
weight = (1300 - x2_mean) / x2_std

co2 = (theta_5[0, 0] + theta_5[1, 0] * eng_size + theta_5[2, 0] * weight) * y_std + y_mean
print("CO2 for 2300 engine size and 1300 weight: ", co2)
