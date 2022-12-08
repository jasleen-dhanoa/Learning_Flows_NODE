from torch import nn

class ODEFunc(nn.Module):
    def __init__(self, device):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 164),
            nn.Tanh(),
            nn.Linear(164, 2),
        ).to(device)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):

        nn_model = self.net(y)
        return nn_model

    #changed model to ReLu instead of Tanh
    #Increased the size of the middle layer from 164 to 512

class M_out(nn.Module):
    def __init__(self, device):
        super(M_out, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 1),
            # nn.ReLU(),
            # nn.Linear(2,1 ),
        ).to(device)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)

    def forward(self, y):
        weights = self.net(y)
        return weights