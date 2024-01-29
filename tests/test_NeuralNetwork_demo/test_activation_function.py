from NeuralNetwork_demo.activation_function import relu_activation, sigmoid_activation
import torch
import torch.nn as nn
import random

def test_relu_activation():
    assert relu_activation([1, 1, 1]) == [1, 1, 1]
    assert relu_activation([1, -1, -1]) == [1, 0, 0]

def test_sigmoid():
    input = [random.random()]
    output = sigmoid_activation(predict=input)
    assert all(0 <= element <= 1 for element in output)

    