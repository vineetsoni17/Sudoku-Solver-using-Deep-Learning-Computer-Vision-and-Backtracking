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
    layers = (nx, 1000, 500, 100, 10)
    j = 1
    for i in os.listdir("Parameters"):
        layer = np.load(f"Parameters/{i}")
        parameters["W"+str(j)] = layer['W']
        parameters["b"+str(j)] = layer['b']
        j += 1
    parameters["X0"] = X_Y_writer.data_X(nx)
    parameters["Y0"] = X_Y_writer.data_Y()
    parameters["X_dev"] = np.load("X_dev.npy")
    parameters["Y_dev"] = np.load("Y_dev.npy")
    parameters["X"] = vectorization(im, parameters)
    forward_propagation(parameters, layers)
    index = np.argmax(parameters["_Y_"], axis=0)
    return index

def vectorization(im, parameters):
    # ret, x = cv2.threshold(im, 160, 255, cv2.THRESH_BINARY_INV)
    x = im
    feature_vector = x.reshape(-1, 1)
    feature_vector = feature_vector.astype('float64')
    mean = np.mean(parameters["X0"])
    feature_vector -= mean
    feature_vector /= np.std(parameters["X0"])
    return feature_vector

def data_shuffle(parameters, m):
    shuffle = np.random.permutation(m)
    a = (parameters["X0"].T[shuffle]).T
    b = (parameters["Y0"].T[shuffle]).T
    parameters["X"] = a[:, :-100]
    np.save("X", parameters["X"])
    parameters["Y"] = b[:, :-100]
    np.save("Y", parameters["Y"])
    parameters["X_dev"] = a[: , -100: ]
    np.save("X_dev", parameters["X_dev"])
    parameters["Y_dev"] = b[: , -100: ]
    np.save("Y_dev", parameters["Y_dev"])
    return parameters

def initialize_parameters(layers, parameters):
    for i in range(1, len(layers)):
        parameters["W"+str(i)] = np.random.randn(layers[i], layers[i-1])*np.sqrt(1/layers[i-1])
        parameters["b"+str(i)] = np.zeros((layers[i],1))
    return parameters

def forward_propagation(parameters, layers):
    parameters["A0"] = parameters["X"]
    for i in range(1, len(layers)):
        parameters["Z"+str(i)] = np.dot(parameters["W"+str(i)], parameters["A"+str(i-1)]) + parameters["b"+str(i)]
        parameters["A"+str(i)] = np.tanh(parameters["Z"+str(i)])
    a = np.exp(parameters["Z"+str(len(layers)-1)])
    parameters["_Y_"] = a/np.sum(a, axis=0)
    return parameters

def cost_function(parameters):
    loss = -(np.multiply(parameters["Y"], np.log(parameters["_Y_"])) + np.multiply(1-parameters["Y"],np.log(1-parameters["_Y_"])))
    cost = np.sum(loss)/parameters["X"].shape[1]
    return cost

def gradient_descent_and_update_parameters(parameters, layers, grads, learning_rate):
    l = len(layers) 
    m = parameters["X"].shape[1]
    grads["dA"+str(len(layers)-1)] = -np.divide(parameters["Y"], parameters["_Y_"]) + np.divide(1-parameters["Y"], 1-parameters["_Y_"])
    for i in range(1,l):
        g = l-i
        if g == l-1:
            grads["dZ"+str(g)] = parameters["_Y_"] - parameters["Y"]
        else:
            grads["dZ"+str(g)] = np.multiply(grads["dA"+str(g)],1/(np.cosh(parameters["Z"+str(g)]**2)))
        grads["dW"+str(g)] = np.dot(grads["dZ"+str(g)], parameters["A"+str(g-1)].T)/m
        grads["db"+str(g)] = np.sum(grads["dZ"+str(g)], axis=1, keepdims=True)/m
        grads["dA"+str(g-1)] = np.dot(parameters["W"+str(g)].T, grads["dZ"+str(g)])
        parameters["W"+str(g)] = parameters["W"+str(g)] - learning_rate*grads["dW"+str(g)]
        parameters["b"+str(g)] = parameters["b"+str(g)] - learning_rate*grads["db"+str(g)]
    return grads, parameters

def actual_answer(Y):
    y = np.where(Y >= np.max(Y), 1, 0)
    return y

def main():
    print("Initializing Program......")

    nx = 2500
    learning_rate = 0.05
    iterations = 10
    parameters = {}
    grads = {}
    parameters["X0"] = X_Y_writer.data_X(nx)
    parameters["Y0"] = X_Y_writer.data_Y()
    layers = (nx, 1000, 500, 100, 10)
    m = len(os.listdir("Database/Dataset"))-1
    cost_lib = []

    data_shuffle(parameters, m)
    initialize_parameters(layers, parameters)
     
    print("Training Neural Network, please wait......")
    # mini batch starting 
    queryX = parameters["X"]
    queryY = parameters["Y"]
    for j in range(0, parameters["X"].shape[1]/81):
        parameters["X"] = query[:, i*queryX.shape[1]/81: (i+1)*queryX.shape[1]/81]
        parameters["Y"] = query[:, i*queryY.shape[1]/81: (i+1)*queryY.shape[1]/81]
    # mini batch starting
        for i in range(0, iterations):
            forward_propagation(parameters, layers)
            cost = cost_function(parameters)
            gradient_descent_and_update_parameters(parameters, layers, grads, learning_rate)
            cost_lib.append(cost)
    
    for i in range(1, len(layers)):
        np.savez("Parameters/Layer_"+str(i), W=parameters["W"+str(i)], b=parameters["b"+str(i)])
    
    print("Traing Successful!")
    plt.plot(range(0, iterations), cost_lib)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost Function")
    plt.show()

    return 0

# main()
    





