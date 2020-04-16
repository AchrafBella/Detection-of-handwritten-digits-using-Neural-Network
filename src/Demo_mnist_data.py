# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 13:49:51 2020

@author: Supernova
"""
from Neural_Network import Neural
import numpy as np

# reseau de neurones 
n = Neural(784, 200, 10,1)

#uplaod data 
training_data_file = open("mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

for record in training_data_list:
    all_values = record.split(',')
    inputs = (np.asfarray(all_values[1:])/255.0*0.99) + 0.01
    targets = np.zeros(10) + 0.01
    targets[int(all_values[0])] = 0.99
    n.fit(inputs, targets,2)
    pass
    
test_data_file = open("mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

#test the network

#scorecard for how well the network performs
scorecard = []

for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    print(correct_label, "correct label")
    inputs = (np.asfarray(all_values[1:])/255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    print(label, "network's answer")
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

#print(scorecard)

scorecard_array = np.asarray(scorecard)
print("performance = ", (scorecard_array.sum() / scorecard_array.size) * 100, "%")