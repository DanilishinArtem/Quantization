import torch
import numpy as np
import torch.nn as nn
# torch.random.manual_seed(1)

class Model(nn.Module):
    def __init__(self, inputSize: int, outputSize: int, hiddenSize: int) -> None:
        super().__init__()
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize = hiddenSize
        self.layer1 = nn.Linear(self.inputSize, self.hiddenSize)
        self.layer2 = nn.Linear(self.hiddenSize, self.hiddenSize)
        self.layer3 = nn.Linear(self.hiddenSize, self.outputSize)
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x



class Generator:
    def __init__(self, nSamples: int, distr: str, distParams: tuple) -> None:
        self.nSamples = nSamples
        self.distParams = distParams
        self.gen = np.random.__getattribute__(distr)
        self.inputDataset = []
        self.outputDataset = []
        self.trueParameters = []
        self.startParameters = []


    def saveParameters(self, model):
        for par in model.parameters():
            self.trueParameters.append(par.data)

    def createNewParameters(self, model):
        for par in model.parameters():
            par.data = torch.tensor(self.gen(*self.distParams, par.data.numel()), requires_grad=True)
            self.startParameters.append(par.data)

    def generateDataset(self, model):
        for i in range(self.nSamples):
            self.inputDataset.append(torch.rand((1,model.inputSize)))
            self.outputDataset.append(model(self.inputDataset[-1]))


class LearningProcess:
    def __init__(self, model: nn.Module, batchSize: int, epochs: int, inputDataset: torch.Tensor, outputDataset: torch.Tensor):
        self.model = model
        self.batchSize = batchSize
        self.epochs = epochs
        self.inputDataset = inputDataset
        self.outputDataset = outputDataset
        self.optimazer = 
    
    def run(self):
        for epoch in range(self.epochs):
            for batch in self.inputDataset:


        
if __name__ == '__main__':
    model = Model(5, 3, 10)
    generator = Generator(100, 'normal', (0,1))
    generator.saveParameters(model)
    generator.generateDataset(model)
    generator.createNewParameters(model)
    print('the end')
    