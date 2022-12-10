import torch
from torch import nn, Tensor
torch.pi = torch.acos(torch.zeros(1)).item() * 2
import numpy as np

class Dynamics(nn.Module):
    def __init__(self):
        super(Dynamics, self).__init__()
        #added new parameters such as epsilon, omega, psi for a general double gyre equation
        self.A = torch.tensor(10.0) #10
        self.s = torch.tensor(50.0)
        self.mu = torch.tensor(0.04)#0.4
        self.epsilon = torch.tensor(0.5)
        self.omega = torch.tensor((2 * torch.pi) / 10)
        self.psi = torch.tensor(0.0)

    def forward(self, t, y):
        # expects y : [n,1,2]
        # expects t : [n]
        #updated functions for a genearl double gyre equation
        f_xt = (self.epsilon * torch.sin(self.omega*t +self.psi)*torch.square(y[:,0]/self.s)) + ((1-2*self.epsilon*torch.sin(self.omega*t +self.psi))*y[:,0]/self.s)
        df_xt = (2*self.epsilon*torch.sin(self.omega*t +self.psi)*y[:, 0]/self.s) + (1-2*self.epsilon*torch.sin(self.omega*t +self.psi))
        xdot = -np.pi * self.A * torch.sin(np.pi*f_xt) * torch.cos(np.pi*y[:, 1]/self.s) - self.mu*y[:, 0]
        ydot = np.pi * self.A * torch.cos(np.pi*f_xt) * torch.sin(np.pi*y[:, 1]/self.s)*df_xt - self.mu*y[:, 1]
        if y.shape[0] == 1:
            return Tensor([xdot, ydot]).unsqueeze(0)
        else:
            return torch.stack([xdot, ydot], 1)
