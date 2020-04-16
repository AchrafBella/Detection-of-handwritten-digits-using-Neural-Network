# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 00:14:14 2020

@author: Supernova
"""
import numpy as np
from matplotlib import pyplot as plt

class Neural_pro():
    def __init__(self, input_nodes, hidden_nodes1, hidden_nodes2, output_nodes, learningRate, activation_function, derive_function):
        #initialize the nueral nodes 
        self.input_nodes  = input_nodes
        self.hidden_nodes1 = hidden_nodes1
        self.hidden_nodes2 = hidden_nodes2
        self.output_nodes = output_nodes
        
        #initializing theweights randomly
        self.weights1 = np.random.normal(0.0, pow(self.hidden_nodes1, -0.5), (self.input_nodes, self.hidden_nodes1))
        self.weights2 = np.random.normal(0.0, pow(self.hidden_nodes2, -0.5), (self.hidden_nodes1, self.hidden_nodes2))
        self.weights3 = np.random.normal(0.0, pow(self.output_nodes, -0.5), (self.hidden_nodes2, self.output_nodes))

        #activation function
        self.activation_function  = activation_function
        self.derive = derive_function
        # learning rate 
        self.learningRate = learningRate
        #list of error 
        self.liste_error = []
        pass
    
    def fit(self, x_train, y_train, epochs):
        x_train = np.array(x_train, ndim = 2)
        y_train = np.array(y_train, ndim = 2)
        
        for e in range(epochs):
            
            input_layer   = x_train
            hidden_layer1 = self.activation_function(np.dot(input_layer, self.weights1))
            hidden_layer2 = self.activation_function(np.dot(hidden_layer1, self.weights2))
            output_layer  = self.activation_function(np.dot(hidden_layer2, self.weights3))
            
            error         = y_train - output_layer
            modification  = error * self.derive(output_layer)
            
            error_hidden2 = np.dot(modification, self.weights3.T)
            modification2 = error_hidden2 * self.derive(hidden_layer2)
            
            error_hidden1 = np.dot(modification2, self.weights2.T)
            modification1 = error_hidden1 * self.derive(hidden_layer1)
            
            self.liste_error.append(np.mean(error))
                  
            self.weights1 += self.learningRate * np.dot(input_layer.T, modification1)
            self.weights2 += self.learningRate * np.dot(hidden_layer1.T, modification2)
            self.weights3 += self.learningRate * np.dot(hidden_layer2.T, modification)
            pass
    
    def query(self, test):
        input_layer   = test
        hidden_layer1 = self.activation_function(np.dot(input_layer, self.weights1))
        hidden_layer2 = self.activation_function(np.dot(input_layer, self.weights2))
        output_layer  = self.activation_function(np.dot(input_layer, self.weights3))
        return output_layer
            
                    