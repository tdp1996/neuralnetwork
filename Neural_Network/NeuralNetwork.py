from linear import Linear
from activation_function import ReLU

class NeuralNetwork(Linear,ReLU):
    def __init__(self):
        super().__init__(self,Linear,ReLU)
        self.fc1 = Linear(3, 3, 0.5, 0.5)
        self.relu1 = ReLU()
        self.fc2 = Linear(3, 1, 0.5, 0.5)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x
    def __call__(self,x):
        self.sample = x
        return self.forward(x)
    
model = NeuralNetwork()
output = model([[1,1,1,1],[1,1,1,1]])
print(output)



