from Neural_Network_class.linear import Linear

def test_initialize_parameters():
    in_features = 3
    out_features = 3
    linear = Linear(in_features=in_features, out_features=out_features)
    weights_layer, bias_layer = linear.initialize_parameters()
    assert len(weights_layer) == out_features
    assert all(len(weights_node)== in_features for weights_node in weights_layer)
    assert len(bias_layer) == len(weights_layer) 