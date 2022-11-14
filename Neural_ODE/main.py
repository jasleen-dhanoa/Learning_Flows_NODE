# This is a script that implements learning of the dynamics of the double gyre system using Neural ODEs.
# It uses the torchdiffeq library (https://github.com/rtqichen/torchdiffeq)
# --------------------------------
# For questions, comments or bugs, feel free to contact KongYao Chee (ckongyao@seas.upenn.edu)
# --------------------------------

import tqdm
import numpy as np
import torch
from torch import nn, optim
from types import SimpleNamespace
import matplotlib.pyplot as plt

from plotting_funcs import create_fig, visualize, visualize_true
from dynamics import Dynamics
from nn_models import ODEFunc
from utils import get_batch

args_dict = {'method': 'rk4',
             'data_size': 1000, # number of data points per trajectory
             'num_traj': 2,     # number of trajectories in the dataset
             'batch_time': 2,
             'niters': 1000,
             'test_freq': 50,
             'viz': True,
             'adjoint': False}
args = SimpleNamespace(**args_dict)

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device          = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initial condition for first trajectory and generate trajectory
true_y01        = torch.tensor([[-20.0, -40.0]]).to(device)
t1              = torch.linspace(0.0, 20.0, args.data_size).to(device)
with torch.no_grad():
    true_y1     = odeint(Dynamics(), true_y01, t1, method='rk4', options=dict(step_size=0.02)).to(device)

# Start from another initial condition and generate another trajectory
true_y02        = torch.tensor([[30.0, -10.0]]).to(device)
t2              = torch.linspace(0.0, 20.0, args.data_size).to(device)
with torch.no_grad():
    true_y2     = odeint(Dynamics(), true_y02, t2, method='rk4', options=dict(step_size=0.02)).to(device)

# Collect both trajectories
true_y          = torch.cat([true_y1.squeeze(), true_y2.squeeze()]).unsqueeze(1)
t               = torch.cat([t1.squeeze(), t2.squeeze()])

# Save data (optional)
true_y_numpy    = true_y.squeeze().cpu().numpy()
t_numpy         = np.expand_dims(t.cpu().numpy(), 1)
data            = np.hstack([t_numpy, true_y_numpy])
with open('Data/double_gyre_data.npy', 'wb') as f:
    np.save(f, data)

if args.viz:
    visualize_true(t1, true_y1, t2, true_y2, device)
    fig, ax_traj, ax_traj2, ax_vecfield = create_fig()

# Create Neural ODE, set optimizer and loss functions
func        = ODEFunc(device).to(device)
optimizer   = optim.Adam(func.parameters(), lr=1e-2)
lossMSE     = nn.MSELoss()

# Do training
for itr in tqdm.tqdm(range(1, args.niters + 1)):
    optimizer.zero_grad()
    batch_y0, batch_t, batch_y  = get_batch(t, true_y, args.data_size, args.num_traj, args.batch_time, device)
    pred_y                      = odeint(func, batch_y0, batch_t, method=args.method, options=dict(step_size=0.02)).to(device)
    loss                        = lossMSE(pred_y, batch_y)
    loss.backward()
    optimizer.step()

    if itr % args.test_freq == 0:
        print('Iter {:4d} | Training Loss {:e}'.format(itr, loss.item()))
        if args.viz:
            with torch.no_grad():
                pred_y1 = odeint(func, true_y01, t1, method=args.method, options=dict(step_size=0.02))
                pred_y2 = odeint(func, true_y02, t2, method=args.method, options=dict(step_size=0.02))
                visualize(itr, t1, t2, true_y1, true_y2, pred_y1, pred_y2, func, fig, ax_traj, ax_traj2, ax_vecfield, device)

plt.show()
