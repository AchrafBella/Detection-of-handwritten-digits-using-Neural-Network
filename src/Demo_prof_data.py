# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 13:57:19 2020

@author: Supernova
"""
from Neural_Network import Neural, Neural_pro
import numpy as np
import scipy.special

#processing data 
#convert function to turn a string to list of int element
def convert(string):
    liste = []
    for elm in string:
        liste.append(int(elm))
    return liste
#upload data for train
training_data_file = open('train.txt', 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
#preprocessing data 
x_train = []
y_train = []
for elm in training_data_list:
    elm = elm.replace('\n','').replace(',','').replace(' ','')
    x = convert(elm)
    x_train.append(x[:48])
    target = np.zeros(10)
    target[x[-1]] = 1
    y_train.append(target)
#upload data for test 
testing_data_file = open('test.txt', 'r')
testing_data_list = testing_data_file.readlines()
testing_data_file.close()
#preprocessing data 
test = []
for elm in testing_data_list:
    elm = elm.replace('\n','').replace(',','').replace(' ','')
    test.append((convert(elm)))
  
#test with activation function 
# Sigmoid functin and his derive
activation_function  = lambda x:scipy.special.expit(x)
derive = lambda x : x*(1.0 - x)    
 
print(' neural network with one hidden layer ')
#learning rate 
alphas = [1, 0.5, 0.2, 0.01, 0.03, 0.05, 0.06, 0.07, 0.08, 0.09, 0.14]

for alpha in alphas:
    print('_'*50)
    print('trainning with alpha:', alpha)
    n = Neural(48, 60, 10, alpha, activation_function, derive)
    n.fit(x_train, y_train, 2000)
    #n.plot_Neural_error()
    cpt = 0
    for elm in test:
        x = elm[-1]
        y = np.argmax(n.query(elm[:48]))
        if( x == y):
            cpt+=1
    print(cpt,"/20")
            
print(' neural network with two hidden layer ')

for alpha in alphas:
    print('_'*50)
    print('trainning with alpha:', alpha)
    n = Neural_pro(48, 60, 15, 10, alpha, activation_function, derive)
    n.fit(x_train, y_train, 2000)
    #n.plot_Neural_error()
    cpt = 0
    for elm in test:
        x = elm[-1]
        y = np.argmax(n.query(elm[:48]))
        if( x == y):
            cpt+=1
    print(cpt,"/20")



























   
