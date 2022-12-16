import tqdm
import numpy as np
import torch
from torch import nn, optim
from types import SimpleNamespace
import matplotlib.pyplot as plt
from plotting_funcs import *
from dynamics import Dynamics
from nn_models import *
from hybrid_nn_models import *
from augmented_nn_models import *
from utils import get_batch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
np.random.seed(0)


# 0. setting up parameters for training
args_dict = {'method': 'rk4',   # solver
             'data_size': 800, # number of data points per trajectory
             'batch_time': 2,   # look forward
             'niters': 5000,   # num of iterations for training
             'test_freq': 50,   # frequency of testing and generating plots
             'viz': True,       # Whether to visualise the data
             'time_steps': 1,  #Trajectory Time Steps
             'adjoint': False,
             'gyre_type': 'single', # 'single' and 'double'
             'num_traj': 1,  # number of trajectories in the dataset # if single gyre with 1 trajectory then 1
             'save_data': False,
             'exp': 'test',  # 'train' and 'test'
             'model_type':'ANODE', # 'NODE', 'KNODE', 'ANODE'
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
model     = Aug_Hybrid(device).to(device)
model.load_state_dict(torch.load('Models/' + plot_path_t + plot_path + '/ANODE/model.pth'))
model.eval()

# 2. creating ground truth - Define the experiment: Single Gyre or Double Gyre
if args.gyre_type == 'single':
 # Generate Ground Truth for Training:
    # 1. Set Initial Condition for trajectory
    # test initial conditions
    true_init_cond_traj_test     =   torch.ones((1000, 1, 2)).to(device)
    for i in range(1000):
        x = np.random.uniform(-48, -1)
        y = np.random.uniform(1, 48)
        true_init_cond_traj_test[i, :, 0], true_init_cond_traj_test[i, :, 1] = x, y

    # plotting initial conditions
    true_init_cond_traj_plot = true_init_cond_traj_test[:15, :, :]

    
    # 2. Generate time steps for trajectory
    true_time_traj_1             =   torch.linspace(0.0, args.time_steps, args.data_size).to(device)
    # 3. Generate "True" Trajectory for n time steps
    with torch.no_grad():
        true_traj_plot             =  odeint(Dynamics(), true_init_cond_traj_plot.squeeze(), true_time_traj_1, method=args.method, options=dict(step_size=0.02)).to(device)
        print("single plot traj shape",true_traj_plot.shape)
        true_traj_test             =  odeint(Dynamics(), true_init_cond_traj_test.squeeze(), true_time_traj_1, method=args.method, options=dict(step_size=0.02)).to(device)
        print("single test traj shape",true_traj_test.shape)

    # 4. Add time decaying Gaussian noise to the trajectory
    # TODO:
    # Change the format of the true_traj below
    true_y                    = true_traj_plot
    t                         = torch.cat([true_time_traj_1.squeeze()])
    traj_lengths              = [true_traj_plot.shape[0]]
    # Setting up visulisation
    if args.viz:
        # 1. Visualize True Trajectory overlaid with  Vector Field
        visualize_true_single_gyre( t, true_y, device, exp=args.exp, model_type =args.model_type, flow_type = plot_path_t)

elif args.gyre_type == 'double':
 # Generate Ground Truth for Training:
    # 1. Set Initial Condition for trajectory
    true_init_cond_traj_plot1    = torch.tensor([[[-20.0, 15.0]], [[-22.0, 10.0]]]).to(device)
    true_init_cond_traj_plot2    = torch.tensor([[[20.0, 15.0]], [[22.0, 10.0]]]).to(device)

    # trajectories for testing
    true_init_cond_traj_test1   =  torch.ones((1000, 1, 2)).to(device)
    true_init_cond_traj_test2   =  torch.ones((1000, 1, 2)).to(device)
    for i in range(1000):
        x = np.random.uniform(5, 45)
        y = np.random.uniform(2, 48)   # try (0, 50)
        true_init_cond_traj_test1[i, :, 0], true_init_cond_traj_test1[i, :, 1] = x, y
        true_init_cond_traj_test2[i, :, 0], true_init_cond_traj_test2[i, :, 1] = -x, y

    true_init_cond_traj_plot1    =  true_init_cond_traj_test1[:15, :, :]
    true_init_cond_traj_plot2    =  true_init_cond_traj_test2[:15, :, :]


    # 2. Generate time steps for trajectory
    true_time_traj_1         = torch.linspace(0.0, args.time_steps, args.data_size).to(device)
    true_time_traj_2         = torch.linspace(0.0, args.time_steps, args.data_size).to(device)
    # 3. Generate "True" Trajectory for n time steps
    with torch.no_grad():
        true_traj_plot1          = odeint(Dynamics(), true_init_cond_traj_plot1.squeeze(), true_time_traj_1, method=args.method,
                             options=dict(step_size=0.02)).to(device)
        print("double plot 1 shape:",true_traj_plot1.shape)
        true_traj_plot2          = odeint(Dynamics(), true_init_cond_traj_plot2.squeeze(), true_time_traj_2, method=args.method,
                             options=dict(step_size=0.02)).to(device)

        true_traj_test1       =  odeint(Dynamics(), true_init_cond_traj_test1.squeeze(), true_time_traj_1, method=args.method,
                             options=dict(step_size=0.02)).to(device)
        print("double test 1 shape:", true_traj_test1.shape)
        true_traj_test2       =  odeint(Dynamics(), true_init_cond_traj_test2.squeeze(), true_time_traj_2, method=args.method,
                             options=dict(step_size=0.02)).to(device)
        
    # 4. Add Gaussian noise to the trajectory
    # TODO
    # 5. Collect both trajectories
    true_y                    = torch.cat([true_traj_plot1.squeeze(), true_traj_plot2.squeeze()]).unsqueeze(1)
    t                         = torch.cat([true_time_traj_1.squeeze(), true_time_traj_2.squeeze()])
    traj_lengths              = [true_traj_plot1.shape[0], true_traj_plot2.shape[0]]
    # 6.
    if args.viz:
        # 1. Visualize True Trajectories overlaid with  Vector Field
        visualize_true_double_gyre(true_time_traj_1 , true_traj_plot1, true_time_traj_2 , true_traj_plot2, device, exp=args.exp, model_type =args.model_type,flow_type = plot_path_t)



# 3. Save visualisations (optional)
if args.viz:
    if args.gyre_type == 'double':
        # 1. Setup create figures: for streamplot and quiverplot
        fig_s,  ax_vecfield_s = create_fig(args.exp, args.gyre_type,args.model_type, cbar=False)
        fig_q, ax_true_vecfield, ax_pred_vecfield , cbar_ax_1, cbar_ax2 = create_fig(args.exp, 'single','ANODE',cbar=True)
    if args.gyre_type == 'single':
        fig_s,  ax_vecfield_s = create_fig(args.exp, args.gyre_type,args.model_type,cbar=False)
        fig_q, ax_true_vecfield, ax_pred_vecfield , cbar_ax_1, cbar_ax2 = create_fig(args.exp, args.gyre_type, args.model_type,cbar=True)



Loss_MSE = nn.MSELoss()
Loss_MAE = nn.L1Loss()
eps = 1e-12
# 4. Predicting
with torch.no_grad():
    # Get Predictions based on single or double gyre:
    # 1. For Single Gyre
    if args.gyre_type == 'single':
        # plot test
        pred_traj_plot = odeint(model, true_init_cond_traj_plot.squeeze(), true_time_traj_1, method=args.method, options=dict(step_size=0.02))
        print("pred plot single shape:", pred_traj_plot.shape)
        visualize_single_gyre_streamplot(0, true_time_traj_1, true_traj_plot, pred_traj_plot, model, fig_s,
                                            None, ax_vecfield_s, device, exp=args.exp, gyre_type=args.gyre_type,model_type =args.model_type,flow_type = plot_path_t)
        visualize_err_vecfield_knode(0, Dynamics(),model, fig_q, ax_true_vecfield, ax_pred_vecfield ,
                                        cbar_ax_1, cbar_ax2, device, exp=args.exp, gyre_type=args.gyre_type,model_type =args.model_type,flow_type = plot_path_t)

        
        # compute test metrics
        pred_traj_test = odeint(model, true_init_cond_traj_test.squeeze(), true_time_traj_1, method=args.method, options=dict(step_size=0.02))
        print("pred test single shape:", pred_traj_test.shape)

        mse_loss = Loss_MSE(pred_traj_test, true_traj_test)
        rmse_loss = torch.sqrt(mse_loss + eps)
        print("RMSE Loss single", rmse_loss.item())

        mae_loss = Loss_MAE(pred_traj_test, true_traj_test)
        print("MAE Loss single", mae_loss.item())


        '''
        ############################## current method ######################################################
        # 1.1 Get Predictions for True Initial Condition and same time variation
        pred_traj_1         = odeint(func, true_init_cond_traj_1 , true_time_traj_1 , method=args.method, options=dict(step_size=0.02))
        # 1.2 Get Predictions using Knowledge based model for True Initial Conditions and same time variation
        knwlge_based_traj_1 = odeint(Dynamics(), true_init_cond_traj_1 , true_time_traj_1, method=args.method,
                                options=dict(step_size=0.02)).to(device)

        total_pred_traj_1 = couple_out(torch.cat([pred_traj_1.unsqueeze(dim=3), knwlge_based_traj_1.unsqueeze(dim=3)], dim=3))
        total_pred_traj_1 = total_pred_traj_1.squeeze(dim=3)

        # 1.3 Visualize Streamplot showing Prediction of both the NN and Knwoledge based model
        visualize_single_gyre_streamplot(itr, true_time_traj_1, true_traj_1, total_pred_traj_1, func,fig_s, ax_traj_s1, ax_vecfield_s, device)
        # 1.4 Visualize the vector fields
        visualize_err_vecfield_knode(itr, Dynamics(),func, fig_q, ax_true_vecfield, ax_pred_vecfield , ax_err_vecfield, cbar_ax_1, cbar_ax2, cbar_ax_3, device,gyre_type =args.gyre_type)
        ###
        #################################################################################################
        '''

    # 2. For Double Gyre
    elif  args.gyre_type == 'double':
        # 2.1 Get Predictions for True Initial Condition and same time variation
        pred_traj_plot1         = odeint(model, true_init_cond_traj_plot1.squeeze() , true_time_traj_1 , method=args.method, options=dict(step_size=0.02))
        print("pred plot double shape:", pred_traj_plot1.shape)
        pred_traj_plot2         = odeint(model, true_init_cond_traj_plot2.squeeze() , true_time_traj_2 , method=args.method, options=dict(step_size=0.02))
        # 2.2 Get Predictions using Knowledge based model for True Initial Conditions and same time variation
        # Not Applicable here
        # 2.3 Visualize Streamplot showing Prediction of both the NN and Knowledge based model
        visualize_double_gyre_streamplot(0, true_time_traj_1, true_time_traj_2, true_traj_plot1, true_traj_plot2,
                                            pred_traj_plot1 , pred_traj_plot2, model,
                    fig_s, None, None, ax_vecfield_s, device, exp=args.exp, gyre_type=args.gyre_type,model_type =args.model_type,flow_type=plot_path_t)
        # 2.5 Visualize the vector field alone
        visualize_err_vecfield_knode(0, Dynamics(),model, fig_q, ax_true_vecfield, ax_pred_vecfield ,
                                        cbar_ax_1, cbar_ax2, device, exp=args.exp, gyre_type=args.gyre_type,model_type=args.model_type,flow_type=plot_path_t)


        # compute test metrics
        pred_traj_test1 = odeint(model, true_init_cond_traj_test1.squeeze(), true_time_traj_1, method=args.method, options=dict(step_size=0.02))
        print("pred test double shape:", pred_traj_test1.shape)
        pred_traj_test2 = odeint(model, true_init_cond_traj_test2.squeeze(), true_time_traj_2, method=args.method, options=dict(step_size=0.02))

        mse_loss1 = Loss_MSE(pred_traj_test1, true_traj_test1)
        mse_loss2 = Loss_MSE(pred_traj_test2, true_traj_test2)
        mse_loss = mse_loss1 + mse_loss2
        rmse_loss = torch.sqrt(mse_loss + eps)
        print("RMSE Loss double", rmse_loss.item())

        mae_loss1 = Loss_MAE(pred_traj_test1, true_traj_test1)
        mae_loss2 = Loss_MAE(pred_traj_test2, true_traj_test2)
        mae_loss = mae_loss1 + mae_loss2
        print("MAE Loss double", mae_loss.item())