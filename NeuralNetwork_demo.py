from typing import Union
from initialize_parameters import initialize_weights, initialize_bias


def batch_processing_forward(batch_input: list[list[Union[float,int]]],
            layers: list[tuple[int]],
            init_weights: Union[float,int],
            init_bias: Union[float,int]) ->list[Union[float,int]]:
    
    batch_predict = []
    try:
        for single_input in batch_input:
            predict = single_forward(single_input, layers, init_weights, init_bias)
            batch_predict.append(predict)
        return batch_predict
    except AssertionError as e:
        raise e
                

def single_forward(input: list[Union[float,int]],
            layers: list[tuple[int]],
            init_weights: Union[float,int],
            init_bias: Union[float,int]) ->list[Union[float,int]]:
      
    try:
        for layer in layers:               
            #initialize parametes
            weights_layer = initialize_weights(layer[0], layer[1], init_weights)
            bias_layer = initialize_bias(layer[1], init_bias)

            pre_act_all_nodes = pre_activation_all_nodes(input, weights_layer, bias_layer)

            #check the number of nodes in the final layer
            if layers[-1][-1] != 1:
                relu = relu_activation(pre_act_all_nodes)
                input = relu
            else:
                input = pre_act_all_nodes
        return input
    except AssertionError as e:
        raise e
    
        
def relu_activation(pre_act_all_nodes:list[float,int]) ->list[float,int]:
    relu_output = [pre_single_act_node if pre_single_act_node > 0  else 0 for pre_single_act_node in pre_act_all_nodes]
    return relu_output


def pre_activation_all_nodes(input: list[Union[float,int]], 
                            weights_layer: list[list[Union[float,int]]],
                            bias_layer: list[Union[float,int]],
                            ) ->list[Union[float,int]]: 
     
    # assert len(weights_layer) == len(bias_layer), "The length of weights_layer and bias_layer should be the same"
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
    input = [[1, 1, 1],[1, 1, 1]]
    layers = [(3,3),(3,1)]
    init_weights = 0.5
    init_bias = 0.5
    predict = batch_processing_forward(batch_input= input, 
                   layers=layers, 
                   init_weights=init_weights,
                   init_bias=init_bias) 
    print(predict)
    



    
    


            




        
