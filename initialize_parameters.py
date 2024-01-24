import random
from typing import Union

#initialize weight
def initialize_weights(in_features, out_features, init_weights=None) ->list[list[Union[float,int]]]:
    if init_weights is not None:
        weights = [[init_weights]* in_features]* out_features
    else:
        weights = [[random.uniform(-1, 1)] * in_features] * out_features
    return weights

def initialize_bias(out_features, init_bias=None) ->list[Union[float,int]]:
    if init_bias is not None:
        bias = [init_bias] * out_features
    else:
        bias = [random.uniform(-1, 1)] * out_features
    return bias