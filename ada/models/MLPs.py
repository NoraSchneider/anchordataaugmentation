import torch
import torch.nn as nn
from typing import List
from ada import ADA

class MLP(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self, input_units: int, hidden_units: int, out_units:int = 1):
    super().__init__()
    self.block_1 = nn.Sequential(nn.Linear(input_units, hidden_units), nn.ReLU())
    self.fclayer = nn.Sequential(nn.Linear(hidden_units, 1))

  def forward(self, x):
    '''
      Forward pass
    '''
    x = self.block_1(x)
    output = self.fclayer(x)
    return output

  def forward_mixup(self, x1, x2, lam=None):
    x1 = self.block_1(x1)
    x2 = self.block_1(x2)
    x = lam * x1 + (1 - lam) * x2
    output = self.fclayer(x)
    return output
  

  def forward_anchor(self, x1, gamma, anchorMatrix):
    x1 = self.block_1(x1)
    x1_til = ADA.transform_pytorch(X=x1, gamma=gamma, anchorMatrix=anchorMatrix)
    output = self.fclayer(x1_til)
    return output

class MLP3Layers(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self, input_units: int, hidden_units: int, out_units:int = 1):
    super().__init__()
    self.block_1 = nn.Sequential(nn.Linear(input_units, hidden_units), nn.ReLU())
    self.block_2 = nn.Sequential(nn.Linear(hidden_units, hidden_units), nn.ReLU())
    self.block_3 = nn.Sequential(nn.Linear(hidden_units, hidden_units), nn.ReLU())
    self.fclayer = nn.Sequential(nn.Linear(hidden_units, 1))

  def forward(self, x):
    '''
      Forward pass
    '''
    x = self.block_1(x)
    x = self.block_2(x)
    x = self.block_3(x)
    output = self.fclayer(x)
    return output

  def forward_mixup(self, x1, x2, lam=None):
    x1 = self.block_1(x1)
    x2 = self.block_1(x2)
    x = lam * x1 + (1 - lam) * x2
    x = self.block_2(x)
    x = self.block_3(x)
    output = self.fclayer(x)
    return output
  

  def forward_anchor(self, x1, gamma, anchorMatrix):
    x1 = self.block_1(x1)
    x1_til = ADA.transform_pytorch(x1, gamma=gamma, anchorMatrix=anchorMatrix)
    x = self.block_2(x1_til)
    x = self.block_3(x)
    output = self.fclayer(x)
    return output