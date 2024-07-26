import numpy as np
from sigmoidGradient import sigmoid, sigmoidGradient
from nnCost import nnCost

def sGD(input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lda, alpha, MaxEpochs):
    Theta1 = np.random.uniform(-0.18, 0.18, size=(hidden_layer_size, input_layer_size + 1))
    Theta2 = np.random.uniform(-0.18, 0.18, size=(num_labels, hidden_layer_size + 1))
    m = X_train.shape[0]
    cost = []
    # transforming y to NN encoding
    y_k = np.zeros((y_train.shape[0], 3))
    for i in range(y_train.shape[0]):
        y_k[i, (y_train[i, 0] - 1)] = 1 
    for _ in range(MaxEpochs):
        Theta1_grad = np.zeros_like(Theta1)
        Theta2_grad = np.zeros_like(Theta2)
        for q in range(X_train.shape[0]): # for each testing sample
            # forward pass
            a_1 = np.vstack(([1], X_train[q].reshape(-1, 1)))
            z_2 = np.dot(Theta1, a_1)
            a_2 = np.vstack(([1], sigmoid(z_2)))
            z_3 = np.dot(Theta2, a_2)
            a_3 = sigmoid(z_3) # 3x1
            
            # backpropagation
            delta_3 = a_3 - (y_k[q, :].reshape(-1, 1)) # 3x1
            delta_2 = (Theta2[:, 1:].T @ delta_3) * sigmoidGradient(z_2) # 40x1
            
            # gradient descent
            Theta1_grad += np.dot(delta_2, a_1.T)
            Theta2_grad += np.dot(delta_3, a_2.T)
            
        # regularization
        Theta1_grad[:, 1:] += lda * Theta1[:, 1:]
        Theta2_grad[:, 1:] += lda * Theta2[:, 1:]
        
        # adjusting theta
        Theta1 -= alpha * Theta1_grad
        Theta2 -= alpha * Theta2_grad
        
        cost.append(nnCost(Theta1, Theta2, X_train, y_train, 3, lda))
    
    
    # iters = np.arange(MaxEpochs)
    # plt.plot(iters, cost)
    # plt.xlabel("Epochs")
    # plt.ylabel("Cost")
    # plt.savefig("output/ps7-4-e-1.png")
    
    return Theta1, Theta2

if __name__ == "__main__":
    from scipy.io import loadmat
    data = loadmat('input/HW7_Data2_full.mat') 
    X = data['X']
    weights = loadmat('input/HW7_weights_3_full.mat')
    Theta1 = weights['Theta1']
    Theta2 = weights['Theta2']
    y_labels = data['y_labels']
    Theta1, Theta2 = sGD(1024, 40, 3, X, y_labels, 0.1, 0.0001, 50)
    
    
    