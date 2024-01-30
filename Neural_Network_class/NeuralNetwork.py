from linear import Linear
from activation_function import ReLU

class NeuralNetwork():
    def __init__(self):
        self.fc1 = Linear(3, 3)
        self.relu = ReLU()
        self.fc2 = Linear(3, 1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    def __call__(self,x):
        self.sample = x
        return self.forward(x)

if __name__ == "__main__":      
    model = NeuralNetwork()




