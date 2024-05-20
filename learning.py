import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from gradientLogger import GradientLogger


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

    def train(self, model, criterion, optimizer, writer, epochs=100, precision='fp16'):
        model = model.cuda()
        scaler = torch.cuda.amp.GradScaler()
        if precision == 'bf16':
            autocast_dtype = torch.bfloat16
        elif precision == 'fp16':
            autocast_dtype = torch.float16
        else:
            autocast_dtype = torch.float32
        for epoch in range(epochs):
            for inputs, outputs in self.dataset:
                model.train()
                optimizer.zero_grad()
                inputs = inputs.cuda()
                outputs = outputs.cuda()
                with torch.cuda.amp.autocast(dtype=autocast_dtype):
                    predictions = model(inputs)
                    loss = criterion(predictions, outputs)
                normOfTrue = self.getNormOfWeights(model, self.trueParameters)
                normOfStart = self.getNormOfWeights(model, self.startParameters)
                print(f'epoch {epoch}, loss = {loss.item():.4f}, normOfTrue = {normOfTrue:.4f}, normOfStart = {normOfStart:.4f}')
                writer.add_scalar('Loss/train', loss.item(), epoch)
                writer.add_scalar('Norm/True', normOfTrue, epoch)
                writer.add_scalar('Norm/Start', normOfStart, epoch)            

                # scaling backward
                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()
                # optimizer.step()
                
                # regular backward
                loss.backward()
                optimizer.step()

                # gradient logging
                self.gradLogger.log_gradients(epoch)
        self.gradLogger.close()
