import numpy as np 
import matplotlib.pyplot as plt
import os
import cv2 
import torch

import X_Y_writer

def real_deal(im):
    nx = 2500
    parameters = {}
    parameters["X0"] = X_Y_writer.data_X(nx)
    parameters["Y0"] = X_Y_writer.data_Y()
    layers = (nx, 500, 10, 10)
    j = 1
    for i in os.listdir("Parameters"):
        layer = np.load(f"Parameters/{i}")
        parameters["W"+str(j)] = layer['W']
        parameters["b"+str(j)] = layer['b']
        j += 1
    parameters["X_dev"] = np.load("X_dev.npy")
    parameters["Y_dev"] = np.load("Y_dev.npy")
    parameters["X"] = vectorization(im, parameters)
    forward_propagation(parameters, layers)
    index = np.argmax(parameters["_Y_"], axis=0)
    return index

def vectorization(im, parameters):
    ret, x = cv2.threshold(im, 160, 255, cv2.THRESH_BINARY_INV)
    feature_vector = x.reshape(-1, 1)
    feature_vector = feature_vector.astype('float64')
    mean = np.mean(parameters["X0"])
    feature_vector -= mean
    feature_vector /= np.std(parameters["X0"])
    return feature_vector

def sigmoid(x):
    a = np.reciprocal(1 + np.exp(-x))
    return a

def data_shuffle(parameters, m):
    shuffle = np.random.permutation(m)
    a = (parameters["X0"].T[shuffle]).T
    b = (parameters["Y0"].T[shuffle]).T
    parameters["X"] = a[:, :-110]
    np.save("X", parameters["X"])
    parameters["Y"] = b[:, :-110]
    np.save("Y", parameters["Y"])
    parameters["X_dev"] = a[: , -110: ]
    np.save("X_dev", parameters["X_dev"])
    parameters["Y_dev"] = b[: , -110: ]
    np.save("Y_dev", parameters["Y_dev"])
    return parameters

def initialize_parameters(layers, parameters, v, s):
    for i in range(1, len(layers)):
        parameters["W"+str(i)] = np.random.randn(layers[i], layers[i-1])*np.sqrt(1/layers[i-1])
        parameters["b"+str(i)] = np.zeros((layers[i],1))
        v["dW"+str(i)] = np.zeros(parameters["W"+str(i)].shape)
        v["db"+str(i)] = np.zeros(parameters["b"+str(i)].shape)
        s["dW"+str(i)] = np.zeros(parameters["W"+str(i)].shape)
        s["db"+str(i)] = np.zeros(parameters["b"+str(i)].shape)
    return parameters, v, s

def forward_propagation(parameters, layers):
    parameters["A0"] = parameters["X"]
    for i in range(1, len(layers)):
        parameters["Z"+str(i)] = np.dot(parameters["W"+str(i)], parameters["A"+str(i-1)]) + parameters["b"+str(i)]
        parameters["A"+str(i)] = np.tanh(parameters["Z"+str(i)])
    a = np.exp(parameters["Z"+str(len(layers)-1)])
    parameters["_Y_"] = a/np.sum(a, axis=0)
    return parameters

def cost_function(parameters):
    loss = -(np.multiply(parameters["Y"], np.log(parameters["_Y_"]))) #- (np.multiply(1-parameters["Y"], np.log(1-parameters["_Y_"])))
    cost = np.sum(loss)/parameters["X"].shape[1]
    return cost

def gradient_descent_and_update_parameters(parameters, layers, grads, learning_rate, v, s, t):
    beta1 = 0.9
    beta2 = 0.99
    e = 10**-8
    l = len(layers) 
    m = parameters["X"].shape[1]
    grads["dA"+str(len(layers)-1)] = -np.divide(parameters["Y"], parameters["_Y_"]) #+ np.divide(1-parameters["Y"], 1-parameters["_Y_"])
    for i in range(1,l):
        g = l-i
        if g == l-1:
            grads["dZ"+str(g)] = parameters["_Y_"] - parameters["Y"]
        else:
            grads["dZ"+str(g)] = np.multiply(grads["dA"+str(g)],np.reciprocal(np.square(np.cosh(parameters["Z"+str(g)]))))   #grads["dA"+str(g)]*sigmoid(parameters["Z"+str(g)])*(1-sigmoid(parameters["Z"+str(g)]))   #np.multiply(grads["dA"+str(g)],np.reciprocal(np.square(np.cosh(parameters["Z"+str(g)]))))
        grads["dW"+str(g)] = np.dot(grads["dZ"+str(g)], parameters["A"+str(g-1)].T)/m
        v["dW"+str(g)] = (beta1*v["dW"+str(g)] + (1-beta1)*grads["dW"+str(g)])  #/(1-(beta1**t))
        s["dW"+str(g)] = (beta2*s["dW"+str(g)] + (1-beta2)*np.square(grads["dW"+str(g)]))  #/(1-(beta2**t))
        grads["db"+str(g)] = np.sum(grads["dZ"+str(g)], axis=1, keepdims=True)/m
        v["db"+str(g)] = (beta1*v["db"+str(g)] + (1-beta1)*grads["db"+str(g)])  #/(1-(beta1**t))
        s["db"+str(g)] = (beta2*s["db"+str(g)] + (1-beta2)*np.square(grads["db"+str(g)]))  #/(1-(beta2**t))
        grads["dA"+str(g-1)] = np.dot(parameters["W"+str(g)].T, grads["dZ"+str(g)])
        # print("x", learning_rate*v["dW"+str(g)]/(np.sqrt(s["dW"+str(g)])+e))
        # print("w", parameters["W"+str(g)])
        parameters["W"+str(g)] = parameters["W"+str(g)] - learning_rate*v["dW"+str(g)]/(np.sqrt(s["dW"+str(g)])+e)   #grads["dW"+str(g)]  #v["dW"+str(g)]  #/(np.sqrt(s["dW"+str(g)])+e)
        # print("w1", parameters["W"+str(g)])
        parameters["b"+str(g)] = parameters["b"+str(g)] - learning_rate*v["db"+str(g)]/(np.sqrt(s["db"+str(g)])+e)   #grads["db"+str(g)]  #v["db"+str(g)]  #/(np.sqrt(s["db"+str(g)])+e)
    return grads, parameters, v, s

def actual_answer(Y):
    y = np.where(Y >= np.max(Y), 1, 0)
    return y

def main():
    print("Initializing Program......")

    nx = 2500
    learning_rate = 0.005
    iterations = 30
    parameters = {}
    v = {}
    s = {}
    grads = {}
    parameters["X0"] = X_Y_writer.data_X(nx)
    parameters["Y0"] = X_Y_writer.data_Y()
    layers = (nx, 500, 10, 10)
    m = len(os.listdir("Database/Dataset"))-1
    cost_lib = []

    data_shuffle(parameters, m)
    initialize_parameters(layers, parameters, v, s)
    
    print("Training Neural Network, please wait......")
    # mini batch starting 
    queryX = parameters["X"]
    queryY = parameters["Y"]
    mini = 256
    z = 1
    for i in range(0, iterations):
        print(i)
        for j in range(0, queryY.shape[1]//mini):
            parameters["X"] = queryX[:, int(j*mini): int((j+1)*mini)]
            parameters["Y"] = queryY[:, int(j*mini): int((j+1)*mini)]
            forward_propagation(parameters, layers)
            cost = cost_function(parameters)
            gradient_descent_and_update_parameters(parameters, layers, grads, learning_rate, v, s, z)
            cost_lib.append(cost)
            z += 1
    
    for i in range(1, len(layers)):
        np.savez("Parameters/Layer_"+str(i), W=parameters["W"+str(i)], b=parameters["b"+str(i)])
    
    print("Training Successful!")
    plt.plot(range(0, len(cost_lib)), cost_lib)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost Function")
    plt.show()

    return 0

# main()
    





