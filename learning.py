import torch
import torch.nn as nn


class Learning:
    def __init__(self, trueParameters: list, startParameters: list, model: nn.Module, inputDataset: list, outputDataset: list, optimizer: torch.optim.Optimizer, criterion: torch.nn.modules.loss._Loss) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.inputDataset = inputDataset
        self.outputDataset = outputDataset
        self.trueParameters = trueParameters
        self.startParameters = startParameters

    def getNormOfWeights(self, model, parameters: list):
        result = 0
        counter = 0
        for par in model.parameters():
            result += torch.norm(par.data.clone() - parameters[counter])
            counter += 1
        return result

    def train(self, model, criterion, optimizer, writer, epochs=100):
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            inputs = torch.stack(self.inputDataset)
            outputs = torch.stack(self.outputDataset)
            predictions = model(inputs)
            loss = criterion(predictions, outputs)
            normOfTrue = self.getNormOfWeights(model, self.trueParameters)
            normOfStart = self.getNormOfWeights(model, self.startParameters)
            print(f'epoch {epoch}, loss = {loss.item():.4f}, normOfTrue = {normOfTrue:.4f}, normOfStart = {normOfStart:.4f}')
            writer.add_scalar('Loss/train', loss.item(), epoch)
            writer.add_scalar('Norm/True', normOfTrue, epoch)
            writer.add_scalar('Norm/Start', normOfStart, epoch)            
            loss.backward()
            optimizer.step()