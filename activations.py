import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_gradient(x):
    return sigmoid(x)*(1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_gradient(x):
    return 1 - tanh(x)**2

def relu(input):
    x = np.array(input)
    x[x < 0] = 0
    return x

def relu_gradient(input):
    x = np.array(input)
    x[x >= 0] = 1
    x[x < 0] = 0
    return x

def param_relu(input, alpha = 0.1):
    x = np.array(input)
    x[x < 0] = alpha*x[x < 0]
    return x

def param_relu_gradient(input, alpha = 0.1):
    x = np.array(input)
    x[x >= 0] = 1
    x[x < 0] = alpha
    return x

def elu(input, alpha = 0.1):
    x = np.array(input)
    x[x < 0] = alpha*(np.exp(x[x < 0]) - 1)
    return x

def elu_gradient(input, alpha = 0.1):
    x = np.array(input)
    x[x >= 0] = 1
    x[x < 0] = elu(x[x < 0], alpha) + alpha
    return x