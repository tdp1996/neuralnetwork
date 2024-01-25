from typing import Union
import random
from activation_function import ReLU

class Linear:
    def __init__(self,in_features: int, out_features: int, init_weight=None, init_bias=None):
        self.in_features = in_features
        self.out_features = out_features
        self.init_weight = init_weight
        self.init_bias = init_bias
        self.weights_layer = self.initialize_weights()
        self.bias_layer = self.initialize_bias()

    def initialize_weights(self) ->list[list[Union[float,int]]]:   
        if self.init_weight is not None:
            return [[self.init_weight] * self.in_features] * self.out_features          
        else:
            return [[random.uniform(-1, 1)] * self.in_features] * self.out_features
    
    def initialize_bias(self) ->list[Union[float,int]]:
        if self.init_bias is not None:
            return [self.init_bias] * self.out_features
        else:
            return [random.uniform(-1, 1)] * self.out_features
    
    def pre_activation_node(self, sample, weight_node, bias_node) ->Union[float,int]:
        assert len(sample) == len(weight_node), "The shape of input and weights should be the same"
        output = sum(a*b for a,b in zip(sample,weight_node)) + bias_node
        return output
    
    def pre_activation_all_nodes(self, sample) ->list[Union[float,int]]:
        pre_act_all_nodes = []    
        for weights_node_i, bias_node_i in zip(self.weights_layer, self.bias_layer):
            pre_act_note_i = self.pre_activation_node(sample, weights_node_i, bias_node_i)
            pre_act_all_nodes.append(pre_act_note_i)  
        return pre_act_all_nodes

    def __call__(self, sample):
        self.sample = sample
        if all(isinstance(sample,list) and isinstance(single_sample,list) for single_sample in sample):
            length_single_sample = len(sample[0])
            assert all(len(single_sample) == length_single_sample for single_sample in sample),"The length of elements must be the same"
            return [self.pre_activation_all_nodes(single_sample) for single_sample in sample]
        else:
            return self.pre_activation_all_nodes(sample)
        
if __name__ == "__main__":
    linear = Linear(3, 3, 0.5, 0.5)
    relu = ReLU()
    output = linear([1,1,1])
    output = relu(output)
    print(output)
    
    
    