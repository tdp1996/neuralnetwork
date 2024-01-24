import random
from typing import Union

def initialize_weights(in_features: int, 
                       out_features: int, 
                       init_weight=None) ->list[list[Union[float,int]]]:
    if init_weight is not None:
        weights = [[init_weight]* in_features]* out_features
    else:
        weights = [[random.uniform(-1, 1)] * in_features] * out_features
    return weights

def initialize_bias(out_features: int, 
                    init_bias=None) ->list[Union[float,int]]:
    if init_bias is not None:
        bias = [init_bias] * out_features
    else:
        bias = [random.uniform(-1, 1)] * out_features
    return bias