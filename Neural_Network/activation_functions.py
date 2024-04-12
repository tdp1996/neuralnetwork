def relu_activation(pre_act_all_nodes: list[float, int]) -> list[float, int]:
    """
    Compute the rectified linear unit (ReLU) activation function for each element in a list.

    Args:
    - pre_act_all_nodes (list[float, int]): A list containing pre-activation values for each node.

    Returns:
    - list[float, int]: A list containing the output of the ReLU activation function for each node.

    Notes:
    - The ReLU activation function returns 0 for any negative input and the input value itself for any non-negative input.
    """
    relu_output = [
        pre_single_act_node if pre_single_act_node > 0 else 0
        for pre_single_act_node in pre_act_all_nodes
    ]
    return relu_output
