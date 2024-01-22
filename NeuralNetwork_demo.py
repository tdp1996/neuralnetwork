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
                            weights: list[list[Union[float,int]]],
                            bias: list[Union[float,int]],
                            num_nodes: int) ->Union[list[float,int],Union[float,int]]: 
     
    pre_act_all_nodes = []
    if num_nodes > 1:
        for weight_node_i, bias_i in zip(weights,bias):
            pre_act_note_i = pre_activation_node_i(input, weight_node_i, bias_i)
            pre_act_all_nodes.append(pre_act_note_i)           
    else:
        pre_act_all_nodes = pre_activation_node_i(input, weights, bias[0])
    return pre_act_all_nodes

def pre_activation_node_i(input: list[Union[float,int]],
                        weights_node_i: list[Union[float,int]],
                        bias_i: Union[float,int]) -> Union[float,int]: 
    output = sum(a*b for a,b in zip(input, weights_node_i)) + bias_i
    return output

weights =  [[[0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5]],
                [0.5, 0.5, 0.5]]               
bias = [[0.5, 0.5, 0.5],[0.5]]
print(forward([1, 1], weights, bias))



    
    


            




        



















