import torch
import numpy as np
import torch.nn as nn
from model import Model
from generator import Generator
from learning import Learning
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from gradientLogger import GradientLogger
from scipy.optimize import minimize
from tqdm import tqdm
# writer = SummaryWriter()

def setExperiment(log_dir: str, inputSize: int=3, outputSize: int=10, hiddenSize: int=10, nSamples: int=1000, epochs: int=100, batch_size: int=32, precisionName: str='fp16', precision: torch.dtype=torch.float16):
    print("learning for " + precisionName)

    model = Model(inputSize=inputSize, outputSize=outputSize, hiddenSize=hiddenSize)
    gradLogger = GradientLogger(model, log_dir + '/' + precisionName + '_' + str(batch_size) + '_grad')
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    generator = Generator(nSamples, 'normal', (0, 1))
    generator.generateDataset(model)
    generator.saveParameters(model)
    generator.createNewParameters(model)
    dataset = TensorDataset(torch.stack(generator.inputDataset), torch.stack(generator.outputDataset))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    writer = SummaryWriter(log_dir + '/' + precisionName + '_' + str(batch_size))
    learner = Learning(generator.trueParameters, generator.startParameters, model, data_loader, optimizer, criterion, gradLogger=gradLogger)
    learner.train(model, criterion, optimizer, writer, epochs=epochs, precisionName=precisionName, precision=precision)
    writer.close()


def optFoo(parameters, model, learner, criterion, optimizer, epochs, precision):
    inputSize = 3
    outputSize = 10
    hiddenSize = 10
    return learner.train_opt(parameters, model, criterion, optimizer, epochs=epochs, precision=precision)

class ProgressMonitor:
    def __init__(self, total_iterations):
        self.progress_bar = tqdm(total=total_iterations)
    
    def update(self, xk):
        self.progress_bar.update(1)

def optParameters(parameters):
    model = Model(inputSize=3, outputSize=10, hiddenSize=10)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    generator = Generator(1000, 'normal', (0, 1))
    generator.generateDataset(model)
    generator.saveParameters(model)
    generator.createNewParameters(model)
    gradLogger = GradientLogger(model, None)
    dataset = TensorDataset(torch.stack(generator.inputDataset), torch.stack(generator.outputDataset))
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    writer = ''
    learner = Learning(generator.trueParameters, generator.startParameters, model, data_loader, optimizer, criterion, gradLogger=gradLogger)
    epochs = 50
    precision = torch.float16

    max_iter = 100
    progress_monitor = ProgressMonitor(max_iter)
    options = {'maxiter': max_iter}
    result = minimize(optFoo, parameters, args=(model, learner, criterion, optimizer, epochs, precision), method='SLSQP', callback=progress_monitor.update, options=options)
    print('result of optimization:')
    print(result)
    # loss = optFoo(parameters, model, learner, criterion, optimizer, epochs, precision)
    # print('total loss: ' + str(loss))


if __name__ == '__main__':

    # log_dir = '/home/adanilishin/Quantization/logs'
    # setExperiment(log_dir, inputSize=3, outputSize=10, hiddenSize=10, nSamples=1000, epochs=50, batch_size=32, precisionName='float32', precision=torch.float32)
    # setExperiment(log_dir, inputSize=3, outputSize=10, hiddenSize=10, nSamples=1000, epochs=50, batch_size=32, precisionName='float16', precision=torch.float16)
    
    # setExperiment(log_dir, inputSize=3, outputSize=10, hiddenSize=10, nSamples=1000, epochs=50, batch_size=64, precisionName='float32', precision=torch.float32)
    # setExperiment(log_dir, inputSize=3, outputSize=10, hiddenSize=10, nSamples=1000, epochs=50, batch_size=64, precisionName='float16', precision=torch.float16)
    
    # setExperiment(log_dir, inputSize=3, outputSize=10, hiddenSize=10, nSamples=1000, epochs=50, batch_size=128, precisionName='float32', precision=torch.float32)
    # setExperiment(log_dir, inputSize=3, outputSize=10, hiddenSize=10, nSamples=1000, epochs=50, batch_size=128, precisionName='float16', precision=torch.float16)


    # parameters = (0.05, 0.1, 0.5, 0.01)
    parameters = np.array([0.05, 0.1, 0.5, 0.01])
    optParameters(parameters)
