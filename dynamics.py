import torch
from torch import nn, Tensor
import numpy as np

class Dynamics(nn.Module):
    def __init__(self):
        super(Dynamics, self).__init__()
        self.A = 10.0
        self.s = 50.0
        self.mu = 0.4

    def forward(self, t, y):
        xdot = -np.pi * self.A * torch.sin(np.pi*y[:, 0]/self.s) * torch.cos(np.pi*y[:, 1]/self.s) - self.mu*y[:, 0]
        ydot = np.pi * self.A * torch.cos(np.pi*y[:, 0]/self.s) * torch.sin(np.pi*y[:, 1]/self.s) - self.mu*y[:, 1]
        if y.shape[0] == 1:
            return Tensor([xdot, ydot]).unsqueeze(0)
        else:
            return torch.stack([xdot, ydot], 1)