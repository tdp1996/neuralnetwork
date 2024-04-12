from typing import Union
from Neural_Network.activation_functions import relu_activation

def forward_batch_processing(
    batch_input: list[list[Union[float, int]]],
    weights_model: list[list[list[Union[float, int]]]],
    bias_model: list[list[Union[float, int]]],
) -> list[list[Union[float, int]]]:
    """
    Perform forward propagation for a batch of input data through the neural network model.

    Args:
    - batch_input (list[list[Union[float, int]]]): List of lists containing input values for each data point in the batch.
    - weights_model (list[list[list[Union[float, int]]]]): List of lists containing weights for each layer of the model.
    - bias_model (list[list[Union[float, int]]]): List of bias values for each layer of the model.

    Returns:
    - list[list[Union[float, int]]]: List of lists containing output values for each data point in the batch.

    Raises:
    - AssertionError: If the lengths of input data points in the batch are not consistent.

    Notes:
    - This function performs forward propagation through the neural network model for a batch of input data.
    - It iterates through each input data point in the batch, applies the forward function to each data point,
      and collects the output predictions into a list.
    """
    
    length = len(batch_input[0])
    assert all(
        len(input) == length for input in batch_input
    ), "The length of elements must be the same"

    batch_predict = []
    for input_values in batch_input:
        predict = forward(input_values, weights_model, bias_model)
        batch_predict.append(predict)
    return batch_predict


def forward(
    input_values: list[Union[float, int]],
    weights_model: list[list[list[Union[float, int]]]],
    bias_model: list[list[Union[float, int]]],
) -> list[Union[float, int]]:
    """
    Perform forward propagation through the neural network model.

    Args:
    - input_values (list[Union[float, int]]): List of input values to the neural network.
    - weights_model (list[list[list[Union[float, int]]]]): List of lists containing weights for each layer of the model.
    - bias_model (list[list[Union[float, int]]]): List of bias values for each layer of the model.

    Returns:
    - list[Union[float, int]]: Output values of the forward pass through the model.

    Raises:
    - AssertionError: If the number of weights layers and bias layers is not the same,
                      or if either the weights_model or bias_model lists are empty.

    Notes:
    - This function performs forward propagation through the neural network model.
    - It iterates through each layer of the model, computing pre-activation values for each layer
      and applying ReLU activation function for all layers except the output layer.
    """

    assert len(weights_model) == len(
        bias_model
    ), """The number of weights layers and bias layers must be the same"""
    assert len(weights_model) > 0 and len(bias_model) > 0
    num_layers = len(weights_model)
    activation = input_values
    for layer_i in range(num_layers):
        pre_act_all_nodes = pre_activation_all_nodes(
            activation, weights_model[layer_i], bias_model[layer_i]
        )
        if layer_i < num_layers - 1:
            activation = relu_activation(pre_act_all_nodes)
    return pre_act_all_nodes


def pre_activation_all_nodes(
    input_values: list[Union[float, int]],
    weights_layer: list[list[Union[float, int]]],
    bias_layer: list[Union[float, int]],
) -> list[Union[float, int]]:
    """
    Compute the pre-activation values for all nodes in a neural network layer.

    Args:
    - input (list[Union[float, int]]): List of input values to the layer.
    - weights_layer (list[list[Union[float, int]]]): List of lists containing weights for each node in the layer.
    - bias_layer (list[Union[float, int]]): List of bias values for each node in the layer.

    Returns:
    - list[Union[float, int]]: List of pre-activation values for all nodes in the layer.

    Raises:
    - AssertionError: If the length of weights_layer and bias_layer is not the same.

    Notes:
    - This function computes the pre-activation values for all nodes in a neural network layer
      using the pre_activation_node_i function for each node.
    """

    assert len(weights_layer) == len(
        bias_layer
    ), "The length of weights_layer and bias_layer should be the same"
    pre_act_all_nodes = []
    for weights_node_i, bias_node_i in zip(weights_layer, bias_layer):
        pre_act_note_i = pre_activation_node_i(input_values, weights_node_i, bias_node_i)
        pre_act_all_nodes.append(pre_act_note_i)

    return pre_act_all_nodes


def pre_activation_node_i(
    input_values: list[Union[float, int]],
    weights_node_i: list[Union[float, int]],
    bias_node_i: Union[float, int],
) -> Union[float, int]:
    """
    Compute the pre-activation value for a given node in a neural network layer.

    Args:
    - input (list[Union[float, int]]): List of input values to the node.
    - weights_node_i (list[Union[float, int]]): List of weights corresponding to the input values for the node.
    - bias_node_i (Union[float, int]): Bias value for the node.

    Returns:
    - Union[float, int]: The pre-activation value computed using the dot product of the input values and weights,
                         and adding the bias.

    Raises:
    - AssertionError: If the length of input and weights_node_i is not the same.

    Notes:
    - The pre-activation value is calculated as the dot product of the input values and weights,
      followed by adding the bias value.
    """
    assert len(input_values) == len(
        weights_node_i
    ), "The length of input and weights_node_i should be the same"
    output = sum(a * b for a, b in zip(input_values, weights_node_i)) + bias_node_i
    return output
