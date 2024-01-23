from typing import Union

def forward(input : Union[list[list[Union[float,int]]],list], 
            weights:list[list[list[Union[float,int]]]], 
            bias: list[list[Union[float,int]]]) ->Union[list,Union[float,int]]:
    predict = []
    if all(isinstance(element,list) for element in input):
        for single_input in input:
            pre_act_all_nodes_1 = pre_activation_all_nodes(single_input, weights[0], bias[0], 3)
            relu_act_output_1 = relu_activation(pre_act_all_nodes_1)
            pre_act_all_nodes_2 = pre_activation_all_nodes(relu_act_output_1, weights[1], bias[1], 1)
            predict.append(pre_act_all_nodes_2)
    else:
        pre_act_all_nodes_1 = pre_activation_all_nodes(input, weights[0], bias[0], 3)
        relu_act_output_1 = relu_activation(pre_act_all_nodes_1)
        predict = pre_activation_all_nodes(relu_act_output_1, weights[1], bias[1], 1)
    return predict

def relu_activation(pre_act_all_nodes:list[float,int]) ->list[float,int]:
    relu_output = [pre_single_act_node if pre_single_act_node > 0  else 0 for pre_single_act_node in pre_act_all_nodes]
    return relu_output

def pre_activation_all_nodes(input: list[Union[float,int]], 
                            weights_layer: list[list[Union[float,int]]],
                            bias_layer: list[Union[float,int]],
                            ) ->list[float,int]: 
     
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
    



    
    


            




        



















