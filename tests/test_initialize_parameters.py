from initialize_parameters import initialize_weights, initialize_bias

def test_initialize_weights():
    in_features = 3
    out_features = 3
    init_weights = 0.5
    expected_output = [[0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5]]
    weights_1 = initialize_weights(in_features=in_features,
                              out_features=out_features,
                              init_weights=init_weights)
    assert weights_1 == expected_output

    weights_2 = initialize_weights(in_features, out_features)
    assert len(weights_2) == out_features
    assert len(weights_2[0]) == in_features

def test_initialize_bias():
    out_features = 3
    init_bias = 0.5
    expected_output = [0.5, 0.5, 0.5]

    bias_1 = initialize_bias(out_features=out_features,
                              init_bias=init_bias)
    
    bias_2 = initialize_bias(out_features=out_features)
                              
    
    assert expected_output == bias_1
    assert len(bias_2) == out_features
