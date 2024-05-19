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