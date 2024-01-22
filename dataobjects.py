from pydantic import BaseModel
from typing import Union


class Tensor(BaseModel):
    value: Union[int, float]
class TensorBatch(BaseModel):
    value: list[list[Tensor]]
class LayerWeight(BaseModel):
    value: list[list[Tensor]]
class LayerBias(BaseModel):
    value: list[Tensor]
    