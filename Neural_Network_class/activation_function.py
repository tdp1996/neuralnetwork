class ReLU():
    def __init__(self):
        pass
    def relu(self,input):
        self.input = input
        if isinstance(input,list):  
            return [pre_single_act_node if pre_single_act_node > 0  else 0 for pre_single_act_node in input]
        else:
            return self.input
    def __call__(self, input):
        self.input = input
        if isinstance(self.input,list):
            return [self.relu(element) for element in self.input]
        else:
            return self.relu(self.input)
        
        
if __name__ == "__main__":
    relu = ReLU()
    output = relu([2,2,2])
    print(output)

