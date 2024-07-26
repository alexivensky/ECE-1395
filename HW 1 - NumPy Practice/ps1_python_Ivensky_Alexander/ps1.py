# Alex Ivensky - 4485057
# ECE 1395
# Homework 1

import numpy as np
import matplotlib.pyplot as plt 
import time

# 3a

sig_x = 0.6
mu_x = 1.5
x = sig_x * np.random.randn(1000000, 1) + mu_x

# 3b

low_z = -1
high_z = 3
z = np.random.uniform(low_z, high_z, (1000000, 1))

# 3c

fig_x = plt.figure()
plt.hist(x, density=True)
plt.title('3c-1')
plt.savefig("output/ps1-3-c-1.png")
plt.close(fig_x)

fig_z = plt.figure()
plt.hist(z, density=True)
plt.title('3c-2')
plt.savefig("output/ps1-3-c-2.png")
plt.close(fig_z)

# 3d

t1 = time.time()
for i in range(x.shape[0]):
    x[i] += 1
t2 = time.time()
print(f"Loop execution time: {t2 - t1}")

# 3e

x -= 1
t1 = time.time()
x += 1
t2 = time.time()
print(f"Numpy function time: {t2 - t1}")

# 3f

y = z[(z > 0) & (z < 1.5)]
print(f"Number of elements retrieved: {np.size(y)}")

# 4a

A = np.array([[2, 1, 3], [2, 6, 8], [6, 8, 18]])
min_val_column = np.min(A, axis=0)
max_val_row = np.max(A, axis=1)
max_value = np.max(A)
sum_columns = np.sum(A, axis=0)
sum_A = np.sum(A)
B = np.multiply(A, A)
print("4a results:")
print(f"Minimum value in each column: \n{min_val_column}")
print(f"Max value in each row: \n{max_val_row}")
print(f"Highest value in A: \n{max_value}")
print(f"Sum of columns: \n{sum_columns}")
print(f"Sum of A: \n{sum_A}")
print(f"B: \n{B}")

# 4b

B = np.array([[1], [3], [5]])
x = np.linalg.inv(A) * B
print(f"Solution for 4b: \n{x}")

# 4c

x1 = np.array([-0.5, 0, 1.5])
x1_l1 = np.linalg.norm(x1, ord=1)
x1_l2 = np.linalg.norm(x1, ord=2)

x2 = np.array([-1, -1, 0])
x2_l1 = np.linalg.norm(x2, ord=1)
x2_l2 = np.linalg.norm(x2, ord=2)

print("4c NumPy calculations:")
print(f"x1_l1: {x1_l1}")
print(f"x1_l2: {x1_l2}")
print(f"x2_l1: {x2_l1}")
print(f"x2_l2: {x2_l2}")

# 5a

X = np.array([[i+1, i+1, i+1] for i in range(10)])
print(f"5a X: \n {X}")
y = np.array([[i+1] for i in range(10)])

# 5b

combined = np.hstack((X, y))
np.random.shuffle(combined)

X_train = combined[0:8, 0:3]
print(f"X_train: \n {X_train}")
X_test = combined[8:10, 0:3]
print(f"X_test: \n {X_test}")

# 5c

y_train = combined[0:8, 3:4]
print(f"y_train: \n {y_train}")
y_test = combined[8:10, 3:4]
print(f"y_test: \n {y_test}")