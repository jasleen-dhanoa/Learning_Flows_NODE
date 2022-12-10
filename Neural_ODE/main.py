import tqdm
import numpy as np
import torch
from torch import nn, optim
from types import SimpleNamespace
import matplotlib.pyplot as plt
from plotting_funcs import *
from dynamics import Dynamics
from nn_models import ODEFunc
from utils import get_batch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 0. setting up parameters for training
args_dict = {'method': 'rk4',   # solver
             'data_size': 1000, # number of data points per trajectory
             'num_traj': 2,     # number of trajectories in the dataset
             'batch_time': 2,   # look forward
             'niters': 10000,   # num of iterations for training
             'test_freq': 50,   # frequency of testing and generating plots
             'viz': True,       # Whether to visualise the data
             'time_steps': 60,  #Trajectory Time Steps
             'adjoint': False,
             'gyre_type': 'single', # 'single' and 'double'
             'save_data': False,
             'debug_level': 1}# debug_level: 0 --> no debugging, debug_level: 1--> quiver plots of trajectories
args = SimpleNamespace(**args_dict)



if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint



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
    # 4. Add Gaussian noise to the trajectory
    # TODO:
    # Change the format of the true_traj below
    true_y                    = torch.cat([true_traj_1 .squeeze()]).unsqueeze(1)
    t                            = torch.cat([true_time_traj_1.squeeze()])
    # Setting up visulisation
    if args.viz:
        # 1. Visualize True Trajectory overlaid with  Vector Field
        visualize_true_single_gyre( t, true_y, device)
        # visualize_true_quiver_single_gyre(true_init_cond_traj_1, true_time_traj_1, device)
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
    # 6.
    if args.viz:
        # 1. Visualize True Trajectories overlaid with  Vector Field
        visualize_true_double_gyre(true_time_traj_1 , true_traj_1, true_time_traj_2 , true_traj_2, device)
        visualize_true_quiver_double_gyre(true_time_traj_1 , true_traj_1, true_time_traj_2 , true_traj_2, device)


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
        fig_q, ax_traj_q1, ax_traj_q2, ax_vecfield_q, cbar_ax = create_fig(args.gyre_type,cbar=True)
    if args.gyre_type == 'single':
        fig_s, ax_traj_s1, ax_vecfield_s = create_fig(args.gyre_type,cbar=False)
        fig_q, ax_traj_q1, ax_vecfield_q, cbar_ax = create_fig(args.gyre_type,cbar=True)




# 4.  Create Neural ODE, set optimizer and loss functions
func        = ODEFunc(device).to(device)
knowledge   = Dynamics().to(device)
optimizer   = optim.Adam(func.parameters(), lr=1e-3)#lr=1e-2 to 1e-3
lossMSE     = nn.MSELoss()


# 5. Do training
training_loss = []
cbar_returned = None
for itr in tqdm.tqdm(range(1, args.niters + 1)):
    # forward pass
    optimizer.zero_grad()
    batch_y0, batch_t, batch_y  = get_batch(t, true_y, args.data_size, args.num_traj-1, args.batch_time, device)
    # knowledge_y                 = odeint(Dynamics(), batch_y0.squeeze(), batch_t, method=args.method, options=dict(step_size=0.02)).to(device)
    pred_y                      = odeint(func, batch_y0, batch_t, method=args.method, options=dict(step_size=0.02)).to(device)
    # Set all the initial value of knowledge to 0
    # knowledge_y[0] = torch.zeros_like(knowledge_y[0])
    # loss                        = lossMSE(pred_y+knowledge_y.unsqueeze(dim=2), batch_y)
    loss = lossMSE(pred_y, batch_y)
    training_loss.append(loss.item())
    loss.backward()
    optimizer.step()

    if itr % args.test_freq == 0:
        print('Iter {:4d} | Training Loss {:e}'.format(itr, loss.item()))
        if args.viz:
            with torch.no_grad():
                # Get Predictions based on single or double gyre:
                # 1. For Single Gyre
                if args.gyre_type == 'single':
                    # 1.1 Get Predictions for True Initial Condition and same time variation
                    pred_traj_1         = odeint(func, true_init_cond_traj_1 , true_time_traj_1 , method=args.method, options=dict(step_size=0.02))
                    # 1.2 Get Predictions using Knowledge based model for True Initial Conditions and same time variation
                    knwlge_based_traj_1 = odeint(Dynamics(), true_init_cond_traj_1 , true_time_traj_1, method=args.method,
                                         options=dict(step_size=0.02)).to(device)
                    knwlge_based_traj_1[0] = torch.zeros_like(knwlge_based_traj_1[0])
                    # 1.3 Visualize Streamplot showing Prediction of both the NN and Knwoledge based model
                    visualize_single_gyre_streamplot(itr, true_time_traj_1, true_traj_1, pred_traj_1 + knwlge_based_traj_1, fumain.pymain.pync,
                              fig_s, ax_traj_s1, ax_traj_s2, ax_vecfield_s, device)
                    # 1.4 Visualize QuiverPlot showing Prediction of both the NN and Knwoledge based model
                    # visualize_single_gyre_quiverplot(itr, true_time_traj_1, true_traj_1, pred_traj_1 + knwlge_based_traj_1, func,
                    #           fig_q, ax_traj_q1, ax_traj_q2, ax_vecfield_q, device)
                    # 1.5 Visualize the vector field alone
                    ###

                # 2. For Double Gyre
                elif  args.gyre_type == 'double':
                    # 2.1 Get Predictions for True Initial Condition and same time variation
                    pred_traj_1         = odeint(func, true_init_cond_traj_1 , true_time_traj_1 , method=args.method, options=dict(step_size=0.02))
                    pred_traj_2         = odeint(func, true_init_cond_traj_2 , true_time_traj_2 , method=args.method, options=dict(step_size=0.02))
                    # 2.2 Get Predictions using Knowledge based model for True Initial Conditions and same time variation
                    knwlge_based_traj_1 = odeint(Dynamics(), true_init_cond_traj_1 , true_time_traj_1, method=args.method, options=dict(step_size=0.02)).to(device)
                    knwlge_based_traj_2 = odeint(Dynamics(), true_init_cond_traj_2, true_time_traj_2, method=args.method, options=dict(step_size=0.02)).to(device)
                    #2.2.1 Set the knowledge for time step 1 to 0
                    knwlge_based_traj_1[0] = torch.zeros_like(knwlge_based_traj_1[0])
                    knwlge_based_traj_2[0] = torch.zeros_like(knwlge_based_traj_2[0])
                    # 2.3 Visualize Streamplot showing Prediction of both the NN and Knowledge based model
                    visualize_double_gyre_streamplot(itr, true_time_traj_1, true_time_traj_2, true_traj_1, true_traj_2,
                                                     pred_traj_1 + knwlge_based_traj_1, pred_traj_2 + knwlge_based_traj_2, func,
                              fig_s, ax_traj_s1, ax_traj_s2, ax_vecfield_s, device)
                    # 2.4 Visualize QuiverPlot showing Prediction of both the NN and Knowledge based model
                    visualize_double_gyre_quiverplot(itr, true_time_traj_1, true_time_traj_2, true_traj_1, true_traj_2,
                                                     pred_traj_1 + knwlge_based_traj_1, pred_traj_2 + knwlge_based_traj_2, func,
                              fig_q, ax_traj_q1, ax_traj_q2, ax_vecfield_q,device)
                    # 2.5 Visualize the vector field alone
                    # visualize_vector_field2(itr, func, fig, ax_traj, ax_traj2, ax_vecfield, device)
plt.figure()
plt.plot(training_loss,np.arange(len(training_loss)), label ='Training Loss')
plt.savefig('Training_Loss'+str(args.gyre_type))
plt.show()
