import torch
from torch import nn, Tensor
import numpy as np
torch.pi = torch.acos(torch.zeros(1)).item() * 2

class Hybrid(nn.Module):
    def __init__(self,device):
        super(Hybrid, self).__init__()
        # added new parameters such as epsilon, omega, psi for a general double gyre equation
        self.A = torch.tensor(10.0)  # 10
        self.s = torch.tensor(50.0)
        self.mu = torch.tensor(0.04)  # 0.4
        self.epsilon = torch.tensor(0.0)
        self.omega = torch.tensor((2 * torch.pi) / 1)
        self.psi = torch.tensor(0.0)

        self.net = nn.Sequential(
            nn.Linear(2, 164),
            nn.Tanh(),
            nn.Linear(164, 2),
        ).to(device)

        self.M_out = nn.Linear(2,1)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)


    def forward(self, t, y):

        f_xt = (self.epsilon * torch.sin(self.omega * t + self.psi) * torch.square(y[:, 0] / self.s)) + (
                    (1 - 2 * self.epsilon * torch.sin(self.omega * t + self.psi)) * y[:, 0] / self.s)
        df_xt = (2 * self.epsilon * torch.sin(self.omega * t + self.psi) * y[:, 0] / self.s) + (
                    1 - 2 * self.epsilon * torch.sin(self.omega * t + self.psi))
        xdot = -np.pi * self.A * torch.sin(np.pi * f_xt) * torch.cos(np.pi * y[:, 1] / self.s) - self.mu * y[:, 0]
        ydot = np.pi * self.A * torch.cos(np.pi * f_xt) * torch.sin(np.pi * y[:, 1] / self.s) * df_xt - self.mu * y[:, 1]

        
        nn_model = self.net(y)


        hybrid_input  = torch.cat([torch.stack([xdot, ydot], 1).unsqueeze(dim=2) , nn_model.unsqueeze(dim =2)], dim=2)
        hybrid_output = self.M_out(hybrid_input).squeeze(dim=2)
        return hybrid_output




