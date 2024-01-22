from typing import Union
import random

class Linear:
    def __init__(self,in_features: int, out_features: int, init_weight=None, init_bias=None):
        self.in_features = in_features
        self.out_features = out_features
        self.init_weight = init_weight
        self.init_bias = init_bias

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
    
    def pre_activation_node(self, sample, weight_node, bias_node):
        output = sum(a*b for a,b in zip(sample,weight_node)) + bias_node
        return output
    
    def pre_activation_all_nodes(self,sample):
        weights = self.initialize_weights()
        bias = self.initialize_bias()       
        if self.out_features > 1:
            return [self.pre_activation_node(sample, weights_node, bias_node) for weights_node,bias_node in zip(weights,bias)]
        else:
            return self.pre_activation_node(sample, weights[0], bias[0])
       
 
    def forward(self):
        if all(isinstance(element,list) and len(element) == self.in_features for element in self.sample):
            return [self.pre_activation_all_nodes(element) for element in self.sample]
        
        elif isinstance(self.sample,list) and len(self.sample) == self.in_features:
            return self.pre_activation_all_nodes(self.sample)
        
        elif isinstance(self.sample,str):
            return self.sample
             
        else:
            return f"Error: mat1 and mat2 shapes cannot be multiplied"


    def __call__(self, sample):
        self.sample = sample
        return self.forward()
    
    
    