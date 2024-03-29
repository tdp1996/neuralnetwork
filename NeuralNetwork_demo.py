from typing import Union

def forward_batch_processing(batch_input :list[list[Union[float,int]]], 
            weights_model:list[list[list[Union[float,int]]]], 
            bias_model: list[list[Union[float,int]]]) ->list[list[Union[float,int]]]:
    
    length = len(batch_input[0])
    assert all(len(input) == length for input in batch_input), "The length of elements must be the same"
    
    batch_predict = []
    for input in batch_input:
        predict = forward(input, weights_model, bias_model)
        batch_predict.append(predict)
    return batch_predict

def forward(input :list[Union[float,int]], 
            weights_model:list[list[list[Union[float,int]]]], 
            bias_model: list[list[Union[float,int]]]) ->list[Union[float,int]]:
    assert len(weights_model) == len(bias_model), """The number of weights layers and bias layers must be the same"""
    assert len(weights_model) > 0 and len(bias_model) > 0
    num_layers = len(weights_model)
    activation = input
    for layer_i in range(num_layers):       
        pre_act_all_nodes = pre_activation_all_nodes(activation, weights_model[layer_i], bias_model[layer_i])
        if layer_i < num_layers - 1:
            activation = relu_activation(pre_act_all_nodes)
    return pre_act_all_nodes

        

def relu_activation(pre_act_all_nodes:list[float,int]) ->list[float,int]:
    relu_output = [pre_single_act_node if pre_single_act_node > 0  else 0 for pre_single_act_node in pre_act_all_nodes]
    return relu_output

def pre_activation_all_nodes(input: list[Union[float,int]], 
                            weights_layer: list[list[Union[float,int]]],
                            bias_layer: list[Union[float,int]],
                            ) ->list[Union[float,int]]: 
     
    assert len(weights_layer) == len(bias_layer), "The length of weights_layer and bias_layer should be the same"
    pre_act_all_nodes = []    
    for weights_node_i, bias_node_i in zip(weights_layer,bias_layer):
        pre_act_note_i = pre_activation_node_i(input, weights_node_i, bias_node_i)
        pre_act_all_nodes.append(pre_act_note_i)          
    
    return pre_act_all_nodes

def pre_activation_node_i(input: list[Union[float,int]],
                        weights_node_i: list[Union[float,int]],
                        bias_node_i: Union[float,int]) -> Union[float,int]: 
    assert len(input) == len(weights_node_i), "The length of input and weights_node_i should be the same"
    output = sum(a*b for a,b in zip(input, weights_node_i)) + bias_node_i
    return output

if __name__ == "__main__":
    weights =  [[[0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5]],
                    [[0.5, 0.5, 0.5]]]               
    bias = [[0.5, 0.5, 0.5],[0.5]]
    print(forward([1, 1, 1], weights, bias))
    



    
    


            




        



















