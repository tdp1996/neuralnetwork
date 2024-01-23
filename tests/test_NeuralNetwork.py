import torch
from torch import nn
import pytest
from NeuralNetwork_demo import pre_activation_node_i, pre_activation_all_nodes ,relu_activation, forward
from utils import convert_to_LayerWeight, convert_to_LayerBias
from dataobjects import Tensor, TensorBatch, LayerWeight, LayerBias
def test_pre_activation_note_i_wrong_shape():
    weights_note_i = [0.5, 0.5, 0.5]
    bias_note_i = 0.5
    with pytest.raises(ValueError, match="The length of input and weights_node_i should be the same"):
        # Call the function that is expected to raise ValueError
        pre_activation_node_i([1, 1], weights_note_i, bias_note_i)

    
def test_pre_activation_note_i():
    weights_note_i = [Tensor(value=0.5) for _ in range(3)]
    bias_note_i = Tensor(value=0.5)
    input_ = [Tensor(value=-1), Tensor(value=1), Tensor(value=1)]
    assert pre_activation_node_i(
        input = input_,
        weights_node_i = weights_note_i,
        bias_i = bias_note_i
        ) == Tensor(value=1)


def test_pre_activation_all_nodes():
    weights =  [[0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5]]
    bias = [0.5, 0.5, 0.5]
    input = [Tensor(value=1) for _ in range(3)]
    layer_weight = convert_to_LayerWeight(weights)
    bias_weight = convert_to_LayerBias(bias)
    
    assert pre_activation_all_nodes(
    input=input,
    layer_weight=layer_weight,
    layer_bias=bias_weight
    ) == [Tensor(value=2) for _ in range(3)]
    

def test_relu_activation():
    input_1 = [Tensor(value=1) for _ in range(3)]
    input_2 = [Tensor(value=1), Tensor(value=-1), Tensor(value=-1)]
    assert relu_activation(input_1) == input_1
    assert relu_activation(input_2) == [Tensor(value=1), Tensor(value=0), Tensor(value=0)]


def test_forward():
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
    model = SimpleNet()
    input_torch_1 = torch.ones(3)
    input_torch_2 = torch.ones(2,3)
    # batch_size = 1
    input_1 = TensorBatch(
        value=[[Tensor(value=1) for _ in range(3)]]
    )
    # batch_size = 2
    input_2 = TensorBatch(
        value=[[Tensor(value=1) for _ in range(3)] for _ in range(2)]
    )
    layer_weight_1 = LayerWeight(
        value=[[Tensor(value=0.5) for _ in range(3)] for _ in range(3)]
    )
    layer_bias_1 = LayerBias(
        value=[Tensor(value=0.5) for _ in range(3)]
    )
    layer_weight_2 = LayerWeight(
        value=[[Tensor(value=0.5) for _ in range(3)]]
    )
    layer_bias_2 = LayerBias(
        value=[Tensor(value=0.5)]
    )

    expected_output_1 = model(input_torch_1).item()
    expected_output_2 = model(input_torch_2)
    assert forward(
        input_batch=input_1,
        model_weights=[layer_weight_1, layer_weight_2],
        model_bias=[layer_bias_1, layer_bias_2]
    ) == [[Tensor(value=expected_output_1)]]
    # assert forward(
    #     input_batch=input_2,
    #     model_weights=[layer_weight_1, layer_weight_2],
    #     model_bias=[layer_bias_1, layer_bias_2]
    # ) == [expected_output_2]

    
   




