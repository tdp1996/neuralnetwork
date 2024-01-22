# import numpy as np
# a = np.array([[1,1],[2,2],[3,3],[4,4]])
# b = np.array([[1,1,1,1],[2,2,2,2]])
# c = np.dot(a,b)

# i = 0
# while i < 6 :
#     print(i)
#     i +=1

# import torch
# ip = torch.ones(2,3)
# print(ip)
# weights =  [[[0.5, 0.5, 0.5],
#                 [0.5, 0.5, 0.5],
#                 [0.5, 0.5, 0.5]],
#                 [0.5, 0.5, 0.5]]

# # Kích thước của ma trận
# rows, cols = 3, 3

# # Khởi tạo ma trận ngẫu nhiên với giá trị trong khoảng [0, 1)
# random_matrix = [[1 for _ in range(cols)] for _ in range(rows)]
# weights_1 = [[[0.5 for _ in range(3)] for _ in range(3)], [0.5 for _ in range(3)]]
# print(weights_1 == weights)

# def foo(a,b):
#     x = a + 1
#     y = b + 2
#     return x, y
# a,b = foo(1,2)
# print(a)
# print(b)

# def initialize_weights(in_features,out_features,init_weight): #->list[list[Union[float,int]]]:
#         if init_weight is True:
#             pass
#         else:
#             weights = [[init_weight for _ in range(in_features)] for _ in range(out_features)]
#         return weights
# print(initialize_weights(3,3,True))

# import random

# def initialize_weights(in_features, out_features, int_weight=None):
#     if int_weight is not None:
#         # Nếu int_weight có giá trị, sử dụng giá trị đó
#         weights = [[int_weight for _ in range(in_features)] for _ in range(out_features)]
#     else:
#         # Ngược lại, sử dụng random.uniform(-1, 1)
#         weights = [[random.uniform(-1, 1) for _ in range(in_features)] for _ in range(out_features)]
#     return weights

# # Sử dụng hàm
# in_features = 3
# out_features = 4

# # Khởi tạo weights với int_weight mặc định (None)
# weights_default = initialize_weights(in_features, out_features)

# # Khởi tạo weights với int_weight được chỉ định
# int_weight_value = 0.5
# weights_with_int_weight = initialize_weights(in_features, out_features, int_weight_value)
# print(weights_with_int_weight)
# class linear():
#     def __init__(self,sample:None):
#         self.sample = sample
#     def foo_1(self):
#         a = self.sample + 1
#         return a
#     def __call__(self):
#         b = self.foo_1() + 1
#         return b

# linear1 = linear(1)
# print(linear1)
# import random
# a = [random.uniform(-1, 1)]
# b = a*3
# print(b)
# def relu_activation(pre_act_all_nodes:list[float,int]) ->list[float,int]:
#     relu_output = [pre_single_act_node if pre_single_act_node > 0  else 0 for pre_single_act_node in pre_act_all_nodes]
#     return relu_output

# print(relu_activation([1,-1,1]))

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
            return [[self.init_weight] * self.in_features for _ in range(self.out_features)]
        else:
            return [[random.uniform(-1, 1) for _ in range(self.in_features)] for _ in range(self.out_features)]
    
    def initialize_bias(self) ->list[Union[float,int]]:
        if self.init_bias is not None:
            return [self.init_bias] * self.out_features
        else:
            return [random.uniform(-1, 1) for _ in range(self.out_features)]
    
    def pre_activation_nodes(self,sample):
        weights = self.initialize_weights()
        bias = self.initialize_bias()
        pre_activation_all_nodes = []
        if sample is not None:
            if self.out_features > 1:
                for weights_node_i, bias_i in zip(weights,bias):
                    pre_act_note_i = sum(a*b for a,b in zip(sample, weights_node_i)) + bias_i
                    pre_activation_all_nodes.append(pre_act_note_i)
            else:
                pre_activation_all_nodes = sum(a*b for a,b in zip(sample, weights[0])) + bias[0]
        return pre_activation_all_nodes
    def forward(self):
        if all(isinstance(element,list) for element in self.sample):
            predict = [self.pre_activation_nodes(element) for element in self.sample ]
        else:
            predict = self.pre_activation_nodes(self.sample)
        return predict
    
    def __call__(self, sample):
        self.sample = sample
        return self.forward()
    
# model =  Linear(3,1, 0.5, 0.5)
# output = model([[1,1,1],[1,1,1]])
# print(output)  



from abc import ABC, abstractmethod

# Định nghĩa một interface
class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

# Lớp cụ thể kế thừa từ interface Shape
class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius ** 2

# Lớp khác kế thừa từ interface Shape
class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

# Hàm sử dụng đa hình với đối tượng của các lớp khác nhau
def print_area(shape):
    print(f"Area: {shape.area()}")

# Tạo các đối tượng từ các lớp con
circle = Circle(radius=5)
rectangle = Rectangle(width=4, height=6)

a = [1,1,1]
b = [1,1]
for i,j in zip(a,b):
    print(i,j)



