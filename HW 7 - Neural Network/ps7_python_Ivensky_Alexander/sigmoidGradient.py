import numpy as np

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

def sigmoidGradient(z):
    return sigmoid(z) * (1 - sigmoid(z))