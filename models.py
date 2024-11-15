import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy

class MLP(nn.Module):

    def __init__(self, input_size=10, output_size=1, hidden_dim = 1024, depth=3):
        super().__init__()

        self.fc_in = nn.Linear(input_size, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_size)
        self.relu = nn.ReLU()

        self.fcs = nn.ModuleList()

        for i in range(depth):
            self.fcs.append(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x):
        x = self.fc_in(x)
        self.relu(x)
        for k in self.fcs:
            x = k(x)
            x = self.relu(x)
        x = self.fc_out(x)

        return x

class Bi_MLP(nn.Module):

    def __init__(self, 
                 input_size1, 
                 input_size2, 
                 hidden_dim1, 
                 hidden_dim2, 
                 output_dim):
        
        super().__init__()
        self.enc1   = MLP(input_size1, hidden_dim1, 128, depth=1)
        self.enc2   = MLP(input_size2, hidden_dim2, 128, depth=1)
        self.outnet = MLP(hidden_dim1+hidden_dim2, output_dim, 128, depth=1)

    def forward(self, x1, x2):
        y1 = self.enc1(x1)
        y2 = self.enc2(x2)
        y  = self.outnet(torch.concat([y1,y2]))
        return y
    

class Tri_MLP(nn.Module):

    def __init__(self, 
                 input_size1, 
                 input_size2, 
                 input_size3,
                 hidden_dim1, 
                 hidden_dim2, 
                 hidden_dim3,
                 output_dim):
        
        super().__init__()
        self.enc1   = MLP(input_size1, hidden_dim1, 128, depth=1)
        self.enc2   = MLP(input_size2, hidden_dim2, 128, depth=1)
        self.enc3   = MLP(input_size3, hidden_dim3, 128, depth=1)
        self.outnet = MLP(hidden_dim1+hidden_dim2+hidden_dim3, output_dim, 128, depth=1)

    def forward(self, x1, x2, x3):
        y1 = self.enc1(x1)
        y2 = self.enc2(x2)
        y3 = self.enc3(x3)
        y  = self.outnet(torch.concat([y1,y2, y3]))
        return y