from typing import Union
from dataobjects import Tensor, TensorBatch, LayerWeight, LayerBias

def forward(input_batch: TensorBatch, 
            model_weights:list[LayerWeight],
            model_bias: list[LayerBias]) ->TensorBatch:
    assert len(model_weights)==len(model_bias)
    num_layers = len(model_weights)
    model_output = TensorBatch(value=[])
    for single_input in input_batch.value:
        for layer_i in range(num_layers):
            pre_activation_all_nodes_ = pre_activation_all_nodes(input=single_input,
                                                                layer_weight=model_weights[layer_i],
                                                                layer_bias=model_bias[layer_i])
            if layer_i != num_layers-1:
                pre_activation_all_nodes_ = relu_activation(pre_activation_all_nodes_)
        model_output.value.append(pre_activation_all_nodes_)
     
    return model_output

def relu_activation(pre_act_all_nodes:list[Tensor]) ->list[Tensor]:
    relu_output = [pre_single_act_node if pre_single_act_node.value > 0  else Tensor(value=0) 
                   for pre_single_act_node in pre_act_all_nodes]
    return relu_output

def pre_activation_all_nodes(input: list[Tensor], 
                            layer_weight: LayerWeight,
                            layer_bias: LayerBias,
                            ) ->list[Tensor]: 
     
    pre_act_all_nodes = []
    bias_dim = len(layer_bias.value)
    assert len(layer_weight.value) == bias_dim
    for i in range(bias_dim):
        pre_act_single_node = pre_activation_node_i(input=input,
                                                  weights_node_i=layer_weight.value[i],
                                                  bias_i=layer_bias.value[i])
        pre_act_all_nodes.append(pre_act_single_node)
    return pre_act_all_nodes

def pre_activation_node_i(input: list[Tensor],
                        weights_node_i: list[Tensor],
                        bias_i: Tensor) -> Tensor:
    if len(input) != len(weights_node_i):
        raise ValueError("The length of input and weights_node_i should be the same")
    else:
        output = sum(a.value*b.value for a,b in zip(input, weights_node_i)) + bias_i.value
    return Tensor(value=output)



    
    


            




        



















