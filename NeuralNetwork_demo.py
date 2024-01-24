from typing import Union
from initialize_parameters import initialize_weights, initialize_bias


def forward_batch_processing(batch_input: list[list[Union[float,int]]],
            layers: list[tuple[int]],
            init_weight: Union[float,int],
            init_bias: Union[float,int]) ->list[Union[float,int]]:
    
    batch_activation = []
    length = len(batch_input[0])
    assert all(len(input) == length for input in batch_input), "The length of elemets must be the same"
    for single_input in batch_input:
        activation = forward(single_input, layers, init_weight, init_bias)
        batch_activation.append(activation)
    return batch_activation

                
def forward(input: list[Union[float,int]],
            layers: list[tuple[int]],
            init_weight: Union[float,int],
            init_bias: Union[float,int]) ->list[Union[float,int]]:

    activation = input
    num_layers = len(layers)
    for i in range(num_layers): 
        pre_act_all_nodes = pre_activation_all_nodes(activation, layers[i], init_weight, init_bias)
        if i < num_layers-1:
            activation = relu_activation(pre_act_all_nodes)
    return pre_act_all_nodes

           
def relu_activation(pre_act_all_nodes:list[float,int]) ->list[float,int]:
    relu_output = [pre_single_act_node if pre_single_act_node > 0  else 0 
                   for pre_single_act_node in pre_act_all_nodes]
    return relu_output


def pre_activation_all_nodes(input: list[Union[float,int]],
                            num_features : tuple[int],
                            init_weight: Union[float,int],
                            init_bias: Union[float,int]
                            ) ->list[Union[float,int]]: 

    #initialize parameters
    assert len(num_features) == 2 , "The length of elements must be 2"
    in_features, out_features = num_features
    weights_layer = initialize_weights(in_features, out_features, init_weight)
    bias_layer = initialize_bias(out_features, init_bias)

    #linear transformation process
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
    input = [1, 1, 1]
    layers = [(3,3),(3,1)]
    init_weights = 0.5
    init_bias = 0.5
    predict = forward(input= input, 
                   layers=layers, 
                   init_weights=init_weights,
                   init_bias=init_bias) 
    print(predict)
    



    
    


            




        
