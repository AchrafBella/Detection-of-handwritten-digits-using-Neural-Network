# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 02:24:51 2020

@author: Supernova
"""
#importing packages 
import numpy as np

#uploading the data
def upload_data():
    liste = []
    with open('train.txt', 'r') as f:
        for line in f:
            line = line.replace(' ','').replace('\n','').replace(',','')
            liste.append(line)
    return liste
 # converting the string into integer 
def convert(string):
    liste = []
    for elm in string:
        liste.append(int(elm))
    return liste
#preprocessing the data into two data x_train and y_train
def preprocessing_data(liste):
    x_train, y_train = [], []
    for elm in liste:
        x = convert(elm)
        x_train.append(x[0:48])
        y_train.append([x[-1]])
    return x_train, y_train

#activation functions to be tested 
def sigmoid(x, derive):
    if(derive == True):
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))
    
def intergral_sigmoid(x, derive):
    if(derive == True):
        return sigmoid(x)
    else:
        return np.ln(1+np.exp(x))
    
#nueral network 
class Neural_network():
    def __init__(self, inputNode, hiddenNode, outputNode, activation, trainning_rate):
        # initializing the model with numbre of nodes 
        self.inputNode  = inputNode
        self.hiddenNode = hiddenNode
        self.outputNode = outputNode
        
        # initializing the weights randomly 
        """
        self.weights1 = np.random.random((self.inputNode, self.hiddenNode)) 
        self.weights2 = np.random.random((self.hiddenNode, self.outputNode))
        """
        self.weights1 = np.random.normal(0.0, pow(self.hiddenNode, -0.5), (self.inputNode, self.hiddenNode)) 
        self.weights2 = np.random.normal(0.0, pow(self.outputNode, -0.5), (self.hiddenNode, self.outputNode))
         
        # activation function 
        self.activation = lambda x,z : activation(x,z)
        
        #trainning rate 
        self.trainning_rate = trainning_rate
        
        
    def trainning(self, x_train, y_train):
        # preparing the data for the neural 
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        
        input_layer  = x_train
        hidden_layer = self.activation(np.dot(input_layer, self.weights1),  False)
        output_layer = self.activation(np.dot(hidden_layer, self.weights2), False)
    
        #error 
        error        = y_train - output_layer
        modification = error * self.activation(output_layer, True)

        error_hidden_layer = np.dot(modification, self.weights2.T)
        modification1 = error_hidden_layer * self.activation(hidden_layer, True)
        
        self.weights1 += self.trainning_rate * np.dot(input_layer.T , modification1)
        self.weights2 += self.trainning_rate * np.dot(hidden_layer.T, modification)
        
        
        print(output_layer[0])
        print(np.argmax(output_layer[0]))
        
        
        
        
#parameters 
epoches = 1

inputNode  = 48
hiddenNode = 40
outputNode = 10


#testing 

if __name__ == '__main__':
    
    x_train, y_train = preprocessing_data(upload_data())
    
    n = Neural_network(inputNode, hiddenNode, outputNode, sigmoid, 0)
    
    n.trainning(x_train, y_train)









