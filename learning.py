import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from gradientLogger import GradientLogger
from scaler import scale_gradients, scale_gradients_opt
import torch.nn.init as init


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
                # scale_gradients(model=model, gradient_logger=self.gradLogger)        

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

    def trainRandom(self, model, criterion, optimizer, writer, epochs=100, precisionName='fp16', precision=torch.float32):
        model = model.cuda()
        for epoch in range(epochs):
            for inputs, outputs in self.dataset:
                model.train()
                optimizer.zero_grad()
                inputs = inputs.cuda()
                outputs = outputs.cuda()
                predictions = model(inputs)
                loss = criterion(predictions, outputs)
                normOfTrue = self.getNormOfWeights(model, self.trueParameters)
                normOfStart = self.getNormOfWeights(model, self.startParameters)
                print(f'epoch {epoch}, loss = {loss.item():.4f}, normOfTrue = {normOfTrue:.4f}, normOfStart = {normOfStart:.4f}')
                writer.add_scalar('Loss/train', loss.item(), epoch)
                writer.add_scalar('Norm/True', normOfTrue, epoch)
                writer.add_scalar('Norm/Start', normOfStart, epoch)
                self.gradLogger.log_gradients(epoch)
                loss.backward()
                optimizer.step()
                parameters = self.getParameters(model)
                self.randomParameters(model, parameters)
        self.gradLogger.close()
        print('parameters of distributions:')
        for name in parameters:
            print(name + ' : ' + str(parameters[name]))
        return parameters

    def randomParameters(self, model, parameters: dict):
        for name, param in model.named_parameters():
            if 'weight' in name:
                init.normal_(param.data, mean=parameters[name][0], std=parameters[name][1])
            elif 'bias' in name:
                init.normal_(param.data, mean=parameters[name][0], std=parameters[name][1])

    def getParameters(self, model):
        parameters = {}
        for name, param in model.named_parameters():
            if 'weight' in name:
                parameters[name] = (torch.mean(torch.tensor(param.data, dtype=torch.float32)).item(), torch.std(torch.tensor(param.data, dtype=torch.float32)).item())
            elif 'bias' in name:
                parameters[name] = (torch.mean(torch.tensor(param.data, dtype=torch.float32)).item(), torch.std(torch.tensor(param.data, dtype=torch.float32)).item())
        return parameters
    
    def testRandom(self, model, parameters: dict, testDataset: torch.utils.data.DataLoader, writer):
        model = model.cuda()
        self.randomParameters(model, parameters)
        model.eval()
        counter = 0
        for inputs, outputs in testDataset:
            counter += 1
            inputs = inputs.cuda()
            outputs = outputs.cuda()
            predictions = model(inputs)
            loss = self.criterion(predictions, outputs)
            writer.add_scalar('Loss/test', loss.item(), counter)

    def test(self, model, testDataset: torch.utils.data.DataLoader, writer):
        model = model.cuda()
        model.eval()
        counter = 0
        for inputs, outputs in testDataset:
            counter += 1
            inputs = inputs.cuda()
            outputs = outputs.cuda()
            predictions = model(inputs)
            loss = self.criterion(predictions, outputs)
            writer.add_scalar('Loss/test', loss.item(), counter)
