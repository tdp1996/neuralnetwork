from torch import nn
from prepare_data.prepare_data_Pytorch import CustomDataset
from torch.utils.data import DataLoader
import torch

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3, 3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 1)
        nn.init.constant_(self.fc1.weight, 0.5)
        nn.init.constant_(self.fc1.bias, 0.5)
        nn.init.constant_(self.fc2.weight, 0.5)
        nn.init.constant_(self.fc2.bias, 0.5)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
model = SimpleNet()
input_tensor_1 = torch.ones(2)
print(input_tensor_1)


