# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 13:59:21 2020

@author: Supernova
"""
# import numpy as np
from numpy import exp, maximum, minimum, tanh, arctan
from math import pi

# Activation function:  sigmoid
def sigmoid_derive(x, derive = False):
    if(derive == True):
        return x*(1-x)
    else:
        return 1/(1+ exp(-x))

# Activation function: Rectified linear unit (ReLU)
# function-in-numpy#comment62519312_32109519
def ReLU(x, derive = False):
    return maximum(minimum(x, 1, x), 0, x)

# Activation function: tangente hyperbolique
def TanH(x, derive = False):
    return tanh(x)

# Activation function: Gaussian
def Gaussian(x, derive = False):
    return ((exp(-(x**2)))/2 + 0.5)

# Activation function: arctangente
def ArcTan(x, derive = False):
    return (1/pi * arctan(x) + 0.5)