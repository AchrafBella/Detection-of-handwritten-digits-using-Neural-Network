# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:27:05 2020

@author: Supernova
"""
import numpy as np
from math import exp, log2

def upload_data(file):
    liste = []
    with open(file,'r') as f:
        for line in f:
            line = line.replace('\n','').replace(' ','')
            liste.append(line)
    return liste

def convertor(string):
    liste = []
    for elm in string:
        liste.append(int(elm))
    return liste
        
def set_data(liste):
    x_train, y_train = [],[]
    for elm in liste:
        x = convertor(elm)
        x_train.append(x[0:48])
        y_train.append(np.array([x[-1]],dtype=np.float64))
    return x_train,y_train

def activation(x, derive = False):
    x = np.nan_to_num(x)
    if(derive == True):
        return 1/(1+np.exp(-x))
    else:
        return np.log(1+np.exp(x))
    

def ReLu_Activation(X):
    X = np.nan_to_num(X)
    return np.maximum(X, 0, X)
    # for x in numpy.nditer(X, op_flags=['readwrite']):
    #    if x < 0: # or x != numpy.NaN:
    #        x[...] = 0.0
    #    else:
    #        x[...] = x
    # return X


def ReLu_Activation_Derivative(X):
    X = np.nan_to_num(X)
    for x in np.nditer(X, op_flags=['readwrite']):
        if x < 0:  # or x != numpy.NaN:
            x[...] = 0.0
        else:
            x[...] = 1.0
    return X

def softmax(X):
    e = np.exp(X - np.amax(X))
    dist = e / np.sum(e) #, axis=1, keepdims=True)
    return dist


                
x_train, y_train = set_data(upload_data('train.txt'))
x_train = np.array(x_train,dtype=np.float64)


generation = 1

np.random.seed(1)
weights1 = 2*np.random.random((48,32))-1
weights2 = 2*np.random.random((32,10))-1


alphas = [1]

for alpha in alphas:
    print('apprentisage avec ',str(alpha))
    for i in range(generation):
        
        input_layer  = x_train
        hidden_layer = ReLu_Activation(np.dot(input_layer, weights1))
        output_layer = softmax(np.dot(hidden_layer, weights2))
        
        error =  output_layer - y_train
        modification = error * ReLu_Activation_Derivative(output_layer)
        
        error_couche_cache = np.dot(modification, weights2.T)
        modification1 = error_couche_cache * ReLu_Activation_Derivative(hidden_layer)
        
        print('moyenne error output_layer',str(np.mean(np.abs(error))))
            
        weights1 -= alpha*np.dot(input_layer.T, modification1) 
        weights2 -= alpha*np.dot(hidden_layer.T, modification) 
    
    print(np.argsort(output_layer[0]))
    print(y_train[0])

    


    
    

