from NeuralNetwork_demo import pre_activation_node_i, pre_activation_all_nodes ,relu_activation, forward
import torch
import pytest
from torch import nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3, 3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 1)
        nn.init.constant_(self.fc1.weight, 0.5)
        nn.init.constant_(self.fc1.bias, 0.5)
        nn.init.constant_(self.fc2.weight, 0.5)
        nn.init.constant_(self.fc2.bias, 0.5)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def test_pre_activation_note_i_wrong_shape():
    weights_note_i = [0.5, 0.5, 0.5]
    bias_note_i = 0.5
    input = [1, 1]
    with pytest.raises(AssertionError, match="The length of input and weights_node_i should be the same"):
        # Call the function that is expected to raise ValueError
        pre_activation_node_i(
        input=input,
        weights_node_i=weights_note_i,
        bias_node_i=bias_note_i
    )

def test_pre_activation_note_i():
    weights_note_i = [0.5, 0.5, 0.5]
    bias_note_i = 0.5
    assert pre_activation_node_i([1, -1, 1], weights_note_i, bias_note_i) == 1

def test_pre_activation_all_nodes_wrong_shape():
    weights_layer =  [[0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5]]  # wrong shape
    bias_layer = [0.5, 0.5, 0.5]
    input = [1, 1, 1]
    with pytest.raises(AssertionError, match="The length of weights_layer and bias_layer should be the same"):
        # Call the function that is expected to raise ValueError
        pre_activation_all_nodes(
        input=input,
        weights_layer=weights_layer,
        bias_layer=bias_layer
    )
   

def test_pre_activation_all_nodes():
    weights_layer =  [[0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5]]
    bias_layer = [0.5, 0.5, 0.5]
    input = [1, 1, 1]
    assert pre_activation_all_nodes(
        input=input,
        weights_layer=weights_layer,
        bias_layer=bias_layer
    ) == [2, 2, 2]
   
    

def test_relu_activation():
    assert relu_activation([1, 1, 1]) == [1, 1, 1]
    assert relu_activation([1, -1, -1]) == [1, 0, 0]


def test_forward():
    model = SimpleNet()
    input_torch_1 = torch.ones(3)

    input = [1, 1, 1]
    weights =  [[[0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5]],
                [[0.5, 0.5, 0.5]]]               
    bias = [[0.5, 0.5, 0.5],[0.5]]
    expected_output = model(input_torch_1).tolist()
    assert forward(input= input, 
                   weights= weights, 
                   bias= bias) == expected_output




