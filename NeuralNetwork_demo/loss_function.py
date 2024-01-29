from NeuralNetwork_demo.activation_function import sigmoid_activation
import math

def binary_cross_entropy(predict: list[float,int], target: list[float,int]) ->list[float,int]:

    assert all(0 <= element <= 1 for element in predict), "Input must be passed through the sigmoid activation function"
    assert len(predict) == len(target), "The length of predict and target must be the same"
    epsilon = 1e-15 
    loss = []
    for predict_element, target_element in zip(predict, target):
        loss_element = - (target_element * math.log(predict_element + epsilon) + (1 - target_element) * math.log(1 - predict_element + epsilon))
        loss.append(loss_element)
    return loss

    