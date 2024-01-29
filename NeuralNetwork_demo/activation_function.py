import math

def relu_activation(pre_act_all_nodes:list[float,int]) ->list[float,int]:
    relu_output = [pre_single_act_node if pre_single_act_node > 0  else 0 
                   for pre_single_act_node in pre_act_all_nodes]
    return relu_output

def sigmoid_activation(predict:list[float,int]) ->list[float,int]:
    return [1 / (1 + math.exp(-element)) for element in predict]

if __name__ == "__main__":
    input = [-6.5, -1, 0.945]
    a = sigmoid_activation(input)
    print(a)
