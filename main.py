import torch
import numpy as np
import torch.nn as nn
from model import Model
from generator import Generator
from learning import Learning
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


if __name__ == '__main__':
    model = Model(inputSize=3, outputSize=10, hiddenSize=10)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    generator = Generator(100, 'normal', (0, 1))
    generator.generateDataset(model)

    generator.saveParameters(model)
    generator.createNewParameters(model)

    learner = Learning(generator.trueParameters, generator.startParameters, model, generator.inputDataset, generator.outputDataset, optimizer, criterion)
    learner.train(model, criterion, optimizer, writer, epochs=1000)
    writer.close()

