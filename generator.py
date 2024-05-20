import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init


class Generator:
    def __init__(self, nSamples: int, distr: str, distParams: tuple) -> None:
        self.nSamples = nSamples
        self.distParams = distParams
        self.gen = np.random.__getattribute__(distr)
        self.inputDataset = []
        self.outputDataset = []
        self.trueParameters = []
        self.startParameters = []

    def saveParameters(self, model):
        self.trueParameters = [par.data.clone() for par in model.parameters()]

    def createNewParameters(self, model):
        for name, param in model.named_parameters():
            if 'weight' in name:
                init.normal_(param.data, mean=0, std=1)
                # init.kaiming_normal_(param.data)
            elif 'bias' in name:
                init.normal_(param.data, mean=0, std=1)
                # init.zeros_(param.data)
        self.startParameters = [par.data.clone() for par in model.parameters()]

    def generateDataset(self, model):
        self.inputDataset = [torch.randn(3) for _ in range(self.nSamples)]
        self.outputDataset = [torch.randn(10) for _ in range(self.nSamples)]
