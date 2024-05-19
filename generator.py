import torch
import numpy as np


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
        for par in model.parameters():
            self.trueParameters.append(par.data)

    def createNewParameters(self, model):
        for par in model.parameters():
            par.data = torch.tensor(self.gen(*self.distParams, par.data.numel()), requires_grad=True)
            self.startParameters.append(par.data)

    def generateDataset(self, model):
        self.inputDataset = [torch.randn(3) for _ in range(self.nSamples)]
        self.outputDataset = [torch.randn(10) for _ in range(self.nSamples)]
