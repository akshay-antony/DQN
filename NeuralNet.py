import torch
import torch.nn as nn
from torch.optim import Adam
import torchvision
import torchvision.transforms as tt
class Network(nn.Module):
    #input as 4*84*84
    def __init__(self,in_size,out_size):
        super().__init__()
        self.network=nn.Sequential(nn.Conv2d(in_size,32,8,stride=4),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(),
                                   nn.Conv2d(32,64,(4,4),2),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.Conv2d(64,64,(3,3),1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.Flatten(),
                                   nn.Linear(3136,512),
                                   nn.ReLU(),
                                   nn.Linear(512,out_size))

    def forward(self, observation):
        return self.network(observation)

    def train_step(self,target,observation,optimizer):
        value=self.forward(observation)
        loss=nn.MSELoss(target,value)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

