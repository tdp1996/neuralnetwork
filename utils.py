from dataobjects import Tensor, TensorBatch, LayerWeight, LayerBias
def convert_to_Tensor(number: int|float) -> Tensor:
    return Tensor(value=number)
def convert_to_TensorBatch(number: list[list[int|float]]) -> TensorBatch:
    return TensorBatch(value=[[convert_to_Tensor(number=number) for number in row] for row in number])
def convert_to_LayerWeight(number: list[list[int|float]]) -> LayerWeight:
    return LayerWeight(value=[[convert_to_Tensor(number=number) for number in row] for row in number])
def convert_to_LayerBias(number: list[int|float]) -> LayerBias:
    return LayerBias(value=[convert_to_Tensor(number=number) for number in number])
