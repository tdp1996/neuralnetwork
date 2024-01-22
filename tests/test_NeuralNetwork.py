from NeuralNetwork_demo import pre_activation_node_i, pre_activation_all_nodes ,relu_activation, forward
import torch
from NeuralNetwork_pytorch import SimpleNet 

def test_pre_activation_note_i():
    weights_note_i = [0.5, 0.5, 0.5]
    bias_note_i = 0.5
    assert pre_activation_node_i([1, 1], weights_note_i, bias_note_i) == 1.5
    assert pre_activation_node_i([1, -1, 1], weights_note_i, bias_note_i) == 1


def test_pre_activation_all_nodes():
    weights_1 =  [[0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5]]
    bias_1 = [0.5, 0.5, 0.5]
    weights_2 = [0.5, 0.5, 0.5]
    bias_2 = [0.5]

    assert pre_activation_all_nodes([1, 1, 1], weights_1, bias_1, 3) == [2, 2, 2]
    assert pre_activation_all_nodes([1, 1, 1], weights_2, bias_2, 1) == 2
    

def test_relu_activation():
    assert relu_activation([1, 1, 1]) == [1, 1, 1]
    assert relu_activation([1, -1, -1]) == [1, 0, 0]


def test_forward():
    model = SimpleNet()
    input_tensor_1 = torch.ones(3)
    input_tensor_2 = torch.ones(2,3)
    weights =  [[[0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5]],
                [0.5, 0.5, 0.5]]               
    bias = [[0.5, 0.5, 0.5],[0.5]]
    expected_output_1 = model(input_tensor_1).item()
    expected_output_2 = model(input_tensor_2)

    assert forward([1, 1, 1], weights, bias) == expected_output_1
    assert forward([[1, 1, 1],[1, 1, 1]], weights, bias)[0] == expected_output_2[0]




