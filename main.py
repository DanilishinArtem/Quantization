import torch
import numpy as np
import torch.nn as nn
from model import Model
from generator import Generator
from learning import Learning


if __name__ == '__main__':
    model = Model(inputSize=3, outputSize=10, hiddenSize=10)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    generator = Generator(100, 'normal', (0, 1))
    generator.generateDataset(model)
    learner = Learning(model, generator.inputDataset, generator.outputDataset, optimizer, criterion)
    learner.train(model, criterion, optimizer, epochs=100)

