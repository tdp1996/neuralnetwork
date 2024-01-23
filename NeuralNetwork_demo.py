from typing import Union

def forward(input :list[Union[float,int]], 
            weights:list[list[list[Union[float,int]]]], 
            bias: list[list[Union[float,int]]]) ->list[Union[float,int]]:
    
    predict = input
    num_layers = len(weights)
    for layer_i in range(num_layers):
        pre_act_all_nodes = pre_activation_all_nodes(predict, weights[layer_i], bias[layer_i])
        if len(weights[num_layers-1]) != 1:
            relu = relu_activation(pre_act_all_nodes)
            predict = relu
        else:
            predict = pre_act_all_nodes
    return predict
        

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
                    [0.5, 0.5, 0.5]]               
    bias = [[0.5, 0.5, 0.5],[0.5]]
    print(forward([1, 1], weights, bias))
    



    
    


            




        



















