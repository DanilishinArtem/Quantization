import torch
import torch.nn as nn
from generator import Generator

class Learning:
    def __init__(self, model: nn.Module, inputDataset: list, outputDataset: list, optimizer: torch.optim.Optimizer, criterion: torch.nn.modules.loss._Loss) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.inputDataset = inputDataset
        self.outputDataset = outputDataset

    def train(self, model, criterion, optimizer, epochs=100):
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            inputs = torch.stack(self.inputDataset)
            outputs = torch.stack(self.outputDataset)
            predictions = model(inputs)
            loss = criterion(predictions, outputs)
            print(f'epoch {epoch}, loss = {loss.item()}')
            loss.backward()
            optimizer.step()