import numpy as np
from computeCost import computeCost

# X_train: m x (n + 1)
# y_train: m x 1
def gradientDescent(X_train, Y_train, alpha, iters):
    m = X_train.shape[0] # number of training samples
    n = X_train.shape[1] - 1 # number of non-bias features
    theta = np.random.rand(n + 1, 1) 
    theta_temp = theta
    cost = np.zeros((iters, 1))

    for i in range(iters): # for each iteration..
        for t in range(n+1): # for each theta..
            sumCalc = 0 # calculate the sum part first
            for j in range(m): # for each training sample..
                sumCalc += (theta_temp.T @ X_train[j, :].T - Y_train[j, :]) * X_train[j, t] # sum = (theta' * X(i) - y(i)) * x_j(i)
            sumCalc = sumCalc * (alpha / m) # sum = sum * alpha * (1 / m)
            theta_temp[t, 0] = theta_temp[t, 0] - sumCalc # theta = theta - above nonsense
        theta = np.copy(theta_temp)  # simultaneous update after iterating through all thetas
        cost[i, 0] = computeCost(X_train, Y_train, theta) # computing cost after each iteration

    return theta, cost


if __name__ == '__main__':
    x0 = np.array([1, 1, 1, 1])
    x1 = np.array([1, 2, 3, 4])
    x2 = np.array([1, 2, 3, 4])
    X = np.hstack((x0.reshape(-1, 1), x1.reshape(-1, 1), x2.reshape(-1, 1)))
    y = np.array([8, 6, 4, 2]).reshape(-1,1)
    theta1 = np.array([0, 1, 0.5]).reshape(-1,1)
    theta2 = np.array([10, -1, -1]).reshape(-1,1)
    theta3 = np.array([3.5, 0, 0]).reshape(-1,1)
    x3 = np.array([[1, 1, 1], [1, 2, 2], [1, 3, 3], [1, 4, 4]])
    print(computeCost(X, y, theta1))
    theta1_calc, cost1_calc = gradientDescent(X, y, 0.0001, 15)
    print("Theta: \n", theta1_calc)
    print("Cost: \n", cost1_calc)