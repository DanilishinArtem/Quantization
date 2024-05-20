import torch
import numpy as np
import torch.nn as nn
from model import Model
from generator import Generator
from learning import Learning
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
# writer = SummaryWriter()


if __name__ == '__main__':
    print("learning for fp32")
    model = Model(inputSize=3, outputSize=10, hiddenSize=10)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    generator = Generator(100, 'normal', (0, 1))
    generator.generateDataset(model)
    generator.saveParameters(model)
    generator.createNewParameters(model)
    dataset = TensorDataset(torch.stack(generator.inputDataset), torch.stack(generator.outputDataset))
    batch_size = 32
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    writer = SummaryWriter(log_dir='/home/adanilishin/Quantization/logs/fp32')
    learner = Learning(generator.trueParameters, generator.startParameters, model, data_loader, optimizer, criterion)
    learner.train(model, criterion, optimizer, writer, epochs=1000, precision='fp32')
    writer.close()

    print("learning for bf16")
    model = Model(inputSize=3, outputSize=10, hiddenSize=10)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    generator = Generator(100, 'normal', (0, 1))
    generator.generateDataset(model)
    generator.saveParameters(model)
    generator.createNewParameters(model)
    dataset = TensorDataset(torch.stack(generator.inputDataset), torch.stack(generator.outputDataset))
    batch_size = 32
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    writer = SummaryWriter(log_dir='/home/adanilishin/Quantization/logs/bf16')
    learner = Learning(generator.trueParameters, generator.startParameters, model, data_loader, optimizer, criterion)
    learner.train(model, criterion, optimizer, writer, epochs=1000, precision='bf16')
    writer.close()
