import tqdm
import numpy as np
import torch
from torch import nn, optim
from types import SimpleNamespace
import matplotlib.pyplot as plt
from plotting_funcs import *
from dynamics import Dynamics
from augmented_nn_models import *
from hybrid_nn_models import *
from utils import get_batch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 0. setting up parameters for training
args_dict = {'method': 'rk4',   # solver
             'data_size': 800, # number of data points per trajectory
             'num_traj': 1,     # number of trajectories in the dataset # if single gyre with 1 trajectory then 1
             'batch_time': 2,   # look forward
             'niters': 5000,   # num of iterations for training
             'test_freq': 50,   # frequency of testing and generating plots
             'viz': True,       # Whether to visualise the data
             'time_steps': 50,  #Trajectory Time Steps
             'adjoint': False,
             'gyre_type': 'single', # 'single' and 'double'
             'save_data': False,
             'model_type': 'ANODE',  # 'NODE', 'KNODE', 'ANODE'
             'flow_type': 'time-invariant',  # 'time-invariant', 'time-variant'
             'debug_level': 1}# debug_level: 0 --> no debugging, debug_level: 1--> quiver plots of trajectories
args = SimpleNamespace(**args_dict)

# parameters tested:
# 1. single gyre: data_size : 800, niters:5000, time_steps:50, num_traj:1,'gyre_type': 'single'
# 2. double gyre: data_size : 800, niters:5000, time_steps:50, num_traj:2, 'gyre_type': 'double'


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

# 1. Define the experiment: Single Gyre or Double Gyre
if args.gyre_type == 'single':
 # Generate Ground Truth for Training:
    # 1. Set Initial Condition for trajectory
    true_init_cond_traj_1        =   torch.tensor([[-49.0, 10]]).to(device)
    # 2. Generate time steps for trajectory
    true_time_traj_1             =   torch.linspace(0.0, args.time_steps, args.data_size).to(device)
    # 3. Generate "True" Trajectory for n time steps
    with torch.no_grad():
        true_traj_1              =  odeint(Dynamics(), true_init_cond_traj_1, true_time_traj_1, method=args.method, options=dict(step_size=0.02)).to(device)
    # 4. Add time decaying Gaussian noise to the trajectory
    # TODO:
    # Change the format of the true_traj below
    true_y                    = torch.cat([true_traj_1 .squeeze()]).unsqueeze(1)
    t                         = torch.cat([true_time_traj_1.squeeze()])
    traj_lengths              = [true_traj_1.shape[0]]
    # Setting up visulisation
    if args.viz:
        # 1. Visualize True Trajectory overlaid with  Vector Field
        visualize_true_single_gyre( t, true_y, device,model_type =args.model_type, flow_type = plot_path_t)

elif args.gyre_type == 'double':
 # Generate Ground Truth for Training:
    # 1. Set Initial Condition for trajectory
    true_init_cond_traj_1    = torch.tensor([[-49.0, 10]]).to(device)
    true_init_cond_traj_2    = torch.tensor([[49.0, 10]]).to(device)
    # 2. Generate time steps for trajectory
    true_time_traj_1         = torch.linspace(0.0, args.time_steps, args.data_size).to(device)
    true_time_traj_2         = torch.linspace(0.0, args.time_steps, args.data_size).to(device)
    # 3. Generate "True" Trajectory for n time steps
    with torch.no_grad():
        true_traj_1          = odeint(Dynamics(), true_init_cond_traj_1, true_time_traj_1, method=args.method,
                             options=dict(step_size=0.02)).to(device)
        true_traj_2          = odeint(Dynamics(), true_init_cond_traj_2, true_time_traj_2, method=args.method,
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
        visualize_true_double_gyre(true_time_traj_1 , true_traj_1, true_time_traj_2 , true_traj_2, device,model_type =args.model_type, flow_type = plot_path_t)


# 2. Save data (optional)
if args.save_data:
    true_y_numpy              = true_y.squeeze().cpu().numpy()
    t_numpy                   = np.expand_dims(t.cpu().numpy(), 1)
    data                      = np.hstack([t_numpy, true_y_numpy])
    with open('Individual_marker_data/marker_1.csv', 'wb') as f:
        np.save(f, data)
    with open('Data/double_gyre_data.npy', 'wb') as f:
        np.save(f, data)


# 3. Save visualisations (optional)
if args.viz:
    if args.gyre_type == 'double':
        # 1. Setup create figures: for streamplot and quiverplot
        fig_s, ax_traj_s1, ax_traj_s2, ax_vecfield_s = create_fig(args.gyre_type,cbar=False)
        fig_q, ax_true_vecfield, ax_pred_vecfield , ax_err_vecfield, cbar_ax_1, cbar_ax2, cbar_ax_3 = create_fig('single',cbar=True)
    if args.gyre_type == 'single':
        fig_s, ax_traj_s1, ax_vecfield_s = create_fig(args.gyre_type,cbar=False)
        fig_q, ax_true_vecfield, ax_pred_vecfield , ax_err_vecfield, cbar_ax_1, cbar_ax2, cbar_ax_3 = create_fig(args.gyre_type,cbar=True)


# 4.  Create Neural ODE, set optimizer and loss functions
hybrid     = Aug_Hybrid(device).to(device)
optim3     = optim.Adam(hybrid.parameters(), lr=1e-3)
lossMSE     = nn.MSELoss()

# 5. Do training
training_loss = []
cbar_returned = None
for itr in tqdm.tqdm(range(1, args.niters + 1)):
    # forward pass
    optim3.zero_grad()
    batch_y0, batch_t, batch_y  = get_batch(t, true_y, traj_lengths, args.data_size, args.num_traj, args.batch_time, device)

    # new method to get output
    # created a hybrid model and used it
    output                   = odeint(hybrid , batch_y0.squeeze(), batch_t, method=args.method, options=dict(step_size=0.02)).to(device)
    ################# current way of getting knowledge and predictions #################################################################
    '''
    knowledge_y                 = odeint(Dynamics(), batch_y0.squeeze(), batch_t, method=args.method, options=dict(step_size=0.02)).to(
        device)
    # knowledge_y[0] = torch.zeros_like(knowledge_y[0])
    pred_y                      = odeint(func, batch_y0, batch_t, method=args.method, options=dict(step_size=0.02)).to(device)

    output                      = couple_out(torch.cat([pred_y.squeeze(dim=2).unsqueeze(dim=3) ,knowledge_y.unsqueeze(dim=3)],dim=3))
    output                      = output.squeeze(dim=3).unsqueeze(dim=2)
    
    '''
    #######################################################################################################################################
    # create a linear layer with biases As M
    # inputs to M are pred_y and knowledge_y
    # outputs are just output_y
    # This helps it to learn M_out

    if args.gyre_type == 'single':
        loss = lossMSE(output.unsqueeze(dim=2), batch_y)
    elif args.gyre_type == 'double':
        loss_1 = lossMSE(pred_y[:,:traj_lengths[0]-1], batch_y[:,:traj_lengths[0]-1])
        loss_2 = lossMSE(pred_y[:,traj_lengths[0]-1:], batch_y[:,traj_lengths[0]-1:])
        loss = loss_1 + loss_2
    training_loss.append(loss.item())
    loss.backward()
    optim3.step()

    if itr % args.test_freq == 0:
        print('Iter {:4d} | Training Loss {:e}'.format(itr, loss.item()))
        if args.viz:
            with torch.no_grad():
                # Get Predictions based on single or double gyre:
                # 1. For Single Gyre
                if args.gyre_type == 'single':
                    pred_traj_1 = odeint(hybrid, true_init_cond_traj_1, true_time_traj_1, method=args.method, options=dict(step_size=0.02))
                    visualize_single_gyre_streamplot(itr, true_time_traj_1, true_traj_1, pred_traj_1, hybrid, fig_s,
                                                     ax_traj_s1, ax_vecfield_s, device,gyre_type=args.gyre_type,model_type =args.model_type,flow_type = plot_path_t)
                    visualize_err_vecfield_knode(itr, Dynamics(),hybrid, fig_q, ax_true_vecfield, ax_pred_vecfield , ax_err_vecfield,
                                                 cbar_ax_1, cbar_ax2, cbar_ax_3, device,gyre_type=args.gyre_type,model_type =args.model_type,flow_type = plot_path_t)

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
                    pred_traj_1         = odeint(func, true_init_cond_traj_1 , true_time_traj_1 , method=args.method, options=dict(step_size=0.02))
                    pred_traj_2         = odeint(func, true_init_cond_traj_2 , true_time_traj_2 , method=args.method, options=dict(step_size=0.02))
                    # 2.2 Get Predictions using Knowledge based model for True Initial Conditions and same time variation
                    # Not Applicable here
                    # 2.3 Visualize Streamplot showing Prediction of both the NN and Knowledge based model
                    visualize_double_gyre_streamplot(itr, true_time_traj_1, true_time_traj_2, true_traj_1, true_traj_2,
                                                     pred_traj_1 , pred_traj_2, func,
                              fig_s, ax_traj_s1, ax_traj_s2, ax_vecfield_s, device,gyre_type=args.gyre_type,model_type =args.model_type,flow_type = plot_path_t)
                    # 2.5 Visualize the vector field alone
                    visualize_err_vecfield(itr, Dynamics(), func, fig_q, ax_true_vecfield, ax_pred_vecfield,
                                           ax_err_vecfield, cbar_ax_1, cbar_ax2, cbar_ax_3, device,gyre_type=args.gyre_type,model_type =args.model_type,flow_type = plot_path_t)

plt.figure()
plt.plot(np.arange(len(training_loss)),training_loss, label ='Training Loss')
plt.savefig('Images/'+ plot_path_t +'Loss_Plots/' + plot_path + '/' + args.model_type + '/training_Loss_' +plot_path_t +str(args.gyre_type)+str(args.model_type))
plt.show()
