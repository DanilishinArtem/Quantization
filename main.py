import torch
import numpy as np
import torch.nn as nn
from model import Model
from generator import Generator
from learning import Learning
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from gradientLogger import GradientLogger
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
    learner.train(model, criterion, optimizer, writer, epochs=epochs, precision=precision)
    writer.close()



if __name__ == '__main__':

    log_dir = '/home/adanilishin/Quantization/logs'
    setExperiment(log_dir, inputSize=3, outputSize=10, hiddenSize=10, nSamples=1000, epochs=50, batch_size=32, precisionName='float32', precision=torch.float32)
    setExperiment(log_dir, inputSize=3, outputSize=10, hiddenSize=10, nSamples=1000, epochs=50, batch_size=32, precisionName='float16', precision=torch.float16)
    setExperiment(log_dir, inputSize=3, outputSize=10, hiddenSize=10, nSamples=1000, epochs=50, batch_size=32, precisionName='bfloat16', precision=torch.bfloat16)
    
    log_dir = '/home/adanilishin/Quantization/logs'
    setExperiment(log_dir, inputSize=3, outputSize=10, hiddenSize=10, nSamples=1000, epochs=50, batch_size=64, precisionName='float32', precision=torch.float32)
    setExperiment(log_dir, inputSize=3, outputSize=10, hiddenSize=10, nSamples=1000, epochs=50, batch_size=64, precisionName='float16', precision=torch.float16)
    setExperiment(log_dir, inputSize=3, outputSize=10, hiddenSize=10, nSamples=1000, epochs=50, batch_size=64, precisionName='bfloat16', precision=torch.bfloat16)
    
    log_dir = '/home/adanilishin/Quantization/logs'
    setExperiment(log_dir, inputSize=3, outputSize=10, hiddenSize=10, nSamples=1000, epochs=50, batch_size=128, precisionName='float32', precision=torch.float32)
    setExperiment(log_dir, inputSize=3, outputSize=10, hiddenSize=10, nSamples=1000, epochs=50, batch_size=128, precisionName='float16', precision=torch.float16)
    setExperiment(log_dir, inputSize=3, outputSize=10, hiddenSize=10, nSamples=1000, epochs=50, batch_size=128, precisionName='bfloat16', precision=torch.bfloat16)
