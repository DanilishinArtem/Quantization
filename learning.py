import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from gradientLogger import GradientLogger
from scaler import scale_gradients, scale_gradients_opt


class Learning:
    def __init__(self, trueParameters: list, startParameters: list, model: nn.Module, dataset: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, criterion: torch.nn.modules.loss._Loss, gradLogger: GradientLogger) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataset = dataset
        self.trueParameters = trueParameters
        self.startParameters = startParameters
        self.gradLogger = gradLogger

    def getNormOfWeights(self, model, parameters: list):
        result = 0
        counter = 0
        for par in model.parameters():
            result += torch.norm(par.data.clone().cpu() - parameters[counter].clone().cpu())
            counter += 1
        return result

    def train(self, model, criterion, optimizer, writer, epochs=100, precisionName='fp16', precision=torch.float32):
        model = model.cuda()
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(epochs):
            for inputs, outputs in self.dataset:
                model.train()
                optimizer.zero_grad()
                inputs = inputs.cuda()
                outputs = outputs.cuda()
                with torch.cuda.amp.autocast(dtype=precision):
                    predictions = model(inputs)
                    loss = criterion(predictions, outputs)
                normOfTrue = self.getNormOfWeights(model, self.trueParameters)
                normOfStart = self.getNormOfWeights(model, self.startParameters)
                print(f'epoch {epoch}, loss = {loss.item():.4f}, normOfTrue = {normOfTrue:.4f}, normOfStart = {normOfStart:.4f}')
                writer.add_scalar('Loss/train', loss.item(), epoch)
                writer.add_scalar('Norm/True', normOfTrue, epoch)
                writer.add_scalar('Norm/Start', normOfStart, epoch)

                self.gradLogger.log_gradients(epoch)
                scale_gradients(model=model, gradient_logger=self.gradLogger)        

                # scaling backward
                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()
                # optimizer.step()
                
                # regular backward
                loss.backward()
                optimizer.step()

                # gradient logging
        self.gradLogger.close()

    def train_opt(self, parameters, model, criterion, optimizer, epochs=100, precision=torch.float32):
        totalLoss = 0
        model = model.cuda()
        for epoch in range(epochs):
            for inputs, outputs in self.dataset:
                model.train()
                optimizer.zero_grad()
                inputs = inputs.cuda()
                outputs = outputs.cuda()
                with torch.cuda.amp.autocast(dtype=precision):
                    predictions = model(inputs)
                    loss = criterion(predictions, outputs)
                    totalLoss += loss.detach().item()
                scale_gradients_opt(parameters, model=model, gradient_logger=self.gradLogger)        
                loss.backward()
                optimizer.step()
        return totalLoss