import tqdm
import numpy as np
import torch
from torch import nn, optim
from types import SimpleNamespace
import matplotlib.pyplot as plt
from plotting_funcs import *
from dynamics import Dynamics
from nn_models import *
from utils import get_batch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 0. setting up parameters for training
args_dict = {'method': 'rk4',   # solver
             'data_size': 800, # number of data points per trajectory
             'batch_time': 2,   # look forward
             'niters': 5000,   # num of iterations for training
             'test_freq': 50,   # frequency of testing and generating plots
             'viz': True,       # Whether to visualise the data
             'time_steps': 3,  #Trajectory Time Steps
             'adjoint': False,
             'gyre_type': 'double', # 'single' and 'double'
             'num_traj': 2,  # number of trajectories in the dataset # if single gyre with 1 trajectory then 1
             'save_data': False,
             'exp': 'test',  # 'train' and 'test'
             'model_type':'NODE', # 'NODE', 'KNODE', 'ANODE'
             'flow_type': 'time-invariant',  # 'time-invariant', 'time-variant'
             'debug_level': 1}# debug_level: 0 --> no debugging, debug_level: 1--> quiver plots of trajectories
args = SimpleNamespace(**args_dict)

if args.gyre_type == 'double':
    args.num_traj = 2
else:
    args.num_traj = 1


if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

if args.gyre_type == 'single':
    plot_path = "Single_Gyre"
if args.gyre_type == 'double':
    plot_path = "Double_Gyre"

if args.flow_type == 'time-invariant':
    plot_path_t = "Inv_"
if args.flow_type == 'time-variant':
    plot_path_t = "Var_"


# 1. loading model
model     = ODEFunc(device).to(device)
model.load_state_dict(torch.load('Models/' + plot_path_t + plot_path + '/NODE/model.pth'))
model.eval()

# 2. creating ground truth - Define the experiment: Single Gyre or Double Gyre
if args.gyre_type == 'single':
 # Generate Ground Truth for Training:
    # 1. Set Initial Condition for trajectory
    true_init_cond_traj_1        =   torch.tensor([[[-20.0, 25.0]], [[-22.0, 10]]]).to(device)
    # 2. Generate time steps for trajectory
    true_time_traj_1             =   torch.linspace(0.0, args.time_steps, args.data_size).to(device)
    # 3. Generate "True" Trajectory for n time steps
    with torch.no_grad():
        true_traj_1              =  odeint(Dynamics(), true_init_cond_traj_1.squeeze(), true_time_traj_1, method=args.method, options=dict(step_size=0.02)).to(device)
    # 4. Add time decaying Gaussian noise to the trajectory
    # TODO:
    # Change the format of the true_traj below
    true_y                    = torch.cat([true_traj_1 .squeeze()])
    t                         = torch.cat([true_time_traj_1.squeeze()])
    traj_lengths              = [true_traj_1.shape[0]]
    # Setting up visulisation
    if args.viz:
        # 1. Visualize True Trajectory overlaid with  Vector Field
        visualize_true_single_gyre( t, true_y, device, exp=args.exp, model_type =args.model_type, flow_type = plot_path_t)

elif args.gyre_type == 'double':
 # Generate Ground Truth for Training:
    # 1. Set Initial Condition for trajectory
    true_init_cond_traj_1    = torch.tensor([[[-20.0, 15.0]], [[-22.0, 10.0]]]).to(device)
    true_init_cond_traj_2    = torch.tensor([[[20.0, 15.0]], [[22.0, 10.0]]]).to(device)
    # 2. Generate time steps for trajectory
    true_time_traj_1         = torch.linspace(0.0, args.time_steps, args.data_size).to(device)
    true_time_traj_2         = torch.linspace(0.0, args.time_steps, args.data_size).to(device)
    # 3. Generate "True" Trajectory for n time steps
    with torch.no_grad():
        true_traj_1          = odeint(Dynamics(), true_init_cond_traj_1.squeeze(), true_time_traj_1, method=args.method,
                             options=dict(step_size=0.02)).to(device)
        true_traj_2          = odeint(Dynamics(), true_init_cond_traj_2.squeeze(), true_time_traj_2, method=args.method,
                             options=dict(step_size=0.02)).to(device)
    # 4. Add Gaussian noise to the trajectory
    # TODO
    # 5. Collect both trajectories
    true_y                    = torch.cat([true_traj_1.squeeze(), true_traj_2.squeeze()]).unsqueeze(1)
    t                         = torch.cat([true_time_traj_1.squeeze(), true_time_traj_2.squeeze()])
    traj_lengths              = [true_traj_1.shape[0], true_traj_2.shape[0]]
    # 6.
    if args.viz:
        # 1. Visualize True Trajectories overlaid with  Vector Field
        visualize_true_double_gyre(true_time_traj_1 , true_traj_1, true_time_traj_2 , true_traj_2, device, exp=args.exp, model_type =args.model_type,flow_type = plot_path_t)



# 3. Save visualisations (optional)
# 3. Save visualisations (optional)
if args.viz:
    if args.gyre_type == 'double':
        # 1. Setup create figures: for streamplot and quiverplot
        fig_s, ax_vecfield_s = create_fig(args.exp,args.gyre_type,args.model_type,cbar=False)
        fig_q, ax_true_vecfield, ax_pred_vecfield ,  cbar_ax_1, cbar_ax2 = create_fig(args.exp,'single',args.model_type,cbar=True)
    if args.gyre_type == 'single':
        fig_s, ax_vecfield_s = create_fig(args.exp, args.gyre_type,args.model_type,cbar=False)
        fig_q, ax_true_vecfield, ax_pred_vecfield , cbar_ax_1, cbar_ax2 = create_fig(args.exp,args.gyre_type,args.model_type,cbar=True)



# 4. Predicting
with torch.no_grad():
    # Get Predictions based on single or double gyre:
    # 1. For Single Gyre
    if args.gyre_type == 'single':
        pred_traj_1 = odeint(model, true_init_cond_traj_1.squeeze(), true_time_traj_1, method=args.method, options=dict(step_size=0.02))
        visualize_single_gyre_streamplot(0, true_time_traj_1, true_traj_1, pred_traj_1, model, fig_s,
                                            None, ax_vecfield_s, device, exp=args.exp, gyre_type=args.gyre_type,model_type =args.model_type,flow_type = plot_path_t)
        visualize_err_vecfield(0, Dynamics(),model, fig_q, ax_true_vecfield, ax_pred_vecfield , None,
                                        cbar_ax_1, cbar_ax2, None, device, exp=args.exp, gyre_type=args.gyre_type,model_type =args.model_type,flow_type = plot_path_t)

    # 2. For Double Gyre
    elif  args.gyre_type == 'double':
        # 2.1 Get Predictions for True Initial Condition and same time variation
        pred_traj_1         = odeint(model, true_init_cond_traj_1.squeeze() , true_time_traj_1 , method=args.method, options=dict(step_size=0.02))
        pred_traj_2         = odeint(model, true_init_cond_traj_2.squeeze() , true_time_traj_2 , method=args.method, options=dict(step_size=0.02))
        # 2.2 Get Predictions using Knowledge based model for True Initial Conditions and same time variation
        # Not Applicable here
        # 2.3 Visualize Streamplot showing Prediction of both the NN and Knowledge based model
        visualize_double_gyre_streamplot(0, true_time_traj_1, true_time_traj_2, true_traj_1, true_traj_2,
                                            pred_traj_1 , pred_traj_2, model,
                    fig_s, None, None, ax_vecfield_s, device, exp=args.exp, gyre_type=args.gyre_type,model_type =args.model_type,flow_type=plot_path_t)
        # 2.5 Visualize the vector field alone
        visualize_err_vecfield(0, Dynamics(),model, fig_q, ax_true_vecfield, ax_pred_vecfield , None,
                                        cbar_ax_1, cbar_ax2,None, device, exp=args.exp, gyre_type=args.gyre_type,model_type=args.model_type,flow_type=plot_path_t)
