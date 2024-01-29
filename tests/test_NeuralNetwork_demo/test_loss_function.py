from NeuralNetwork_demo.loss_function import binary_cross_entropy
import random
from NeuralNetwork_demo.activation_function import sigmoid_activation
import pytest


def test_binary_cross_entropy():
    input = [random.random()] * 3
    target = [random.random()] * 3
    predict = sigmoid_activation(input)
    output = binary_cross_entropy(predict=predict, target=target)
    assert all(0 <= element <= 1 for element in output)

def test_binary_cross_entropy_error():
    predict = [random.random()] * 2
    predict_1 = [1.5, -0.99, 1]
    target = [random.random()] * 3
    with pytest.raises(AssertionError, match="The length of predict and target must be the same"):
        binary_cross_entropy(predict=predict, target=target)

    with pytest.raises(AssertionError, match="Input must be passed through the sigmoid activation function"):
        binary_cross_entropy(predict=predict_1, target=target)