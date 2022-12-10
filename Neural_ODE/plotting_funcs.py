import matplotlib.pyplot as plt
import numpy as np
import torch
from dynamics import Dynamics
from mpl_toolkits.axes_grid1 import make_axes_locatable

def visualize_true(t1, true_y1, t2, true_y2, device):
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_traj2 = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)

    ax_traj.cla()
    ax_traj.plot(t1.cpu().numpy(), true_y1.cpu().numpy()[:, 0, 0], 'b', label='true x')
    ax_traj.plot(t1.cpu().numpy(), true_y1.cpu().numpy()[:, 0, 1], 'g', label='true y')
    ax_traj.set_xlim(t1.cpu().min(), t1.cpu().max())
    ax_traj.set_ylim(-60, 60)
    ax_traj.set_title('Trajectories (left gyre)')
    ax_traj.set_xlabel('t')
    ax_traj.set_ylabel('x,y')
    ax_traj.legend()
    ax_traj.grid('on')

    ax_traj2.cla()
    ax_traj2.plot(t2.cpu().numpy(), true_y2.cpu().numpy()[:, 0, 0], 'b', label='true x')
    ax_traj2.plot(t2.cpu().numpy(), true_y2.cpu().numpy()[:, 0, 1], 'g', label='true y')
    ax_traj2.set_xlim(t2.cpu().min(), t2.cpu().max())
    ax_traj2.set_ylim(-50, 50)
    ax_traj2.set_title('Trajectories (right gyre)')
    ax_traj2.set_xlabel('t2')
    ax_traj2.set_ylabel('x2,y2')
    ax_traj2.legend()
    ax_traj2.grid()

    y, x = np.mgrid[-60:60:2000j, -50:50:2000j]
    grid_samples = torch.Tensor(np.stack([x, y], -1).reshape(2000 * 2000, 2)).to(device)
    dynamics = Dynamics()
    dydt = dynamics.forward(0, grid_samples).cpu().detach().numpy()
    dydt = dydt.reshape(2000, 2000, 2)
    ax_vecfield.plot(true_y1.cpu().numpy()[:, 0, 0], true_y1.cpu().numpy()[:, 0, 1], 'b', label='true1')
    ax_vecfield.plot(true_y2.cpu().numpy()[:, 0, 0], true_y2.cpu().numpy()[:, 0, 1], 'g', label='true2')
    ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
    ax_vecfield.set_xlim(-50, 50)
    ax_vecfield.set_ylim(-60, 60)
    ax_vecfield.set_title('Vector Field')
    ax_vecfield.set_xlabel('x')
    ax_vecfield.set_ylabel('y')


    fig.tight_layout()
    plt.draw()

def visualize_true_quiver(t1, true_y1, t2, true_y2, device):
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_traj2 = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)

    ax_traj.cla()
    ax_traj.plot(t1.cpu().numpy(), true_y1.cpu().numpy()[:, 0, 0], 'b', label='true x')
    ax_traj.plot(t1.cpu().numpy(), true_y1.cpu().numpy()[:, 0, 1], 'g', label='true y')
    ax_traj.set_xlim(t1.cpu().min(), t1.cpu().max())
    ax_traj.set_ylim(-60, 60)
    ax_traj.set_title('Trajectories (left gyre)')
    ax_traj.set_xlabel('t')
    ax_traj.set_ylabel('x,y')
    ax_traj.legend()
    ax_traj.grid('on')

    ax_traj2.cla()
    ax_traj2.plot(t2.cpu().numpy(), true_y2.cpu().numpy()[:, 0, 0], 'b', label='true x')
    ax_traj2.plot(t2.cpu().numpy(), true_y2.cpu().numpy()[:, 0, 1], 'g', label='true y')
    ax_traj2.set_xlim(t2.cpu().min(), t2.cpu().max())
    ax_traj2.set_ylim(-50, 50)
    ax_traj2.set_title('Trajectories (right gyre)')
    ax_traj2.set_xlabel('t2')
    ax_traj2.set_ylabel('x2,y2')
    ax_traj2.legend()
    ax_traj2.grid()

    res = 100
    y, x = np.mgrid[-60:60:20j, -50:50:20j]
    grid_samples = torch.Tensor(np.stack([x, y], -1).reshape(20 * 20, 2)).to(device)
    dynamics = Dynamics()
    dydt = dynamics.forward(0, grid_samples).cpu().detach().numpy()
    dydt = dydt.reshape(20, 20, 2)
    ax_vecfield.plot(true_y1.cpu().numpy()[:, 0, 0], true_y1.cpu().numpy()[:, 0, 1], 'b', label='true1')
    ax_vecfield.plot(true_y2.cpu().numpy()[:, 0, 0], true_y2.cpu().numpy()[:, 0, 1], 'g', label='true2')
    ax_vecfield.quiver(x, y, dydt[:, :, 0], dydt[:, :, 1])
    ax_vecfield.set_xlim(-50, 50)
    ax_vecfield.set_ylim(-60, 60)
    ax_vecfield.set_title('Vector Field')
    ax_vecfield.set_xlabel('x')
    ax_vecfield.set_ylabel('y')


    fig.tight_layout()
    plt.draw()

def create_fig(gyre_type, model_type, cbar):

    if gyre_type == "double":
        fig = plt.figure(figsize=(12, 4), facecolor='white')
        ax_traj = fig.add_subplot(131, frameon=False)
        ax_traj2 = fig.add_subplot(132, frameon=False)
        ax_vecfield = fig.add_subplot(133, frameon=False)
        if cbar:
            divider = make_axes_locatable(ax_vecfield)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.show(block=False)
            return fig, ax_traj, ax_traj2, ax_vecfield, cax
        plt.show(block=False)
        return fig, ax_traj, ax_traj2, ax_vecfield

    if gyre_type == "single":
        if model_type == "KNODE":
            if cbar:
                fig = plt.figure(figsize=(12, 4), facecolor='white')
                ax_true_vecfield = fig.add_subplot(121, frameon=False)
                ax_pred_vecfield = fig.add_subplot(122, frameon=False)
                divider_1 = make_axes_locatable(ax_true_vecfield )
                cax_1 = divider_1.append_axes('right', size='5%', pad=0.05)
                divider_2 = make_axes_locatable(ax_pred_vecfield )
                cax_2 = divider_2.append_axes('right', size='5%', pad=0.05)
                plt.show(block=False)
                return fig, ax_true_vecfield, ax_pred_vecfield, cax_1, cax_2
            else:
                fig = plt.figure(figsize=(8, 4), facecolor='white')
                ax_traj = fig.add_subplot(121, frameon=False)
                ax_vecfield = fig.add_subplot(122, frameon=False)
                plt.show(block=False)
                return fig, ax_traj,ax_vecfield

        else:
            if cbar:
                fig = plt.figure(figsize=(12, 4), facecolor='white')
                ax_true_vecfield = fig.add_subplot(131, frameon=False)
                ax_pred_vecfield = fig.add_subplot(132, frameon=False)
                ax_err_vecfield = fig.add_subplot(133, frameon=False)
                divider_1 = make_axes_locatable(ax_true_vecfield )
                cax_1 = divider_1.append_axes('right', size='5%', pad=0.05)
                divider_2 = make_axes_locatable(ax_pred_vecfield )
                cax_2 = divider_2.append_axes('right', size='5%', pad=0.05)
                divider_3 = make_axes_locatable( ax_err_vecfield )
                cax_3 = divider_3.append_axes('right', size='5%', pad=0.05)
                plt.show(block=False)
                return fig, ax_true_vecfield, ax_pred_vecfield, ax_err_vecfield, cax_1, cax_2, cax_3
            else:
                fig = plt.figure(figsize=(8, 4), facecolor='white')
                ax_traj = fig.add_subplot(121, frameon=False)
                ax_vecfield = fig.add_subplot(122, frameon=False)
                plt.show(block=False)
                return fig, ax_traj,ax_vecfield





def create_fig_many():
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_traj2 = fig.add_subplot(132, frameon=False)
    ax_traj3 = fig.add_subplot(133, frameon=False)
    ax_vecfield = fig.add_subplot(134, frameon=False)
    plt.show(block=False)

    return fig, ax_traj, ax_traj2, ax_traj3, ax_vecfield
def visualize(itr, t1, t2, true_y1, true_y2, pred_y1, pred_y2, odefunc, fig, ax_traj, ax_traj2, ax_vecfield, device):
    ax_traj.cla()
    ax_traj.plot(t1.cpu().numpy(), true_y1.cpu().numpy()[:, 0, 0], 'b', label='true x')
    ax_traj.plot(t1.cpu().numpy(), true_y1.cpu().numpy()[:, 0, 1], 'g', label='true y')
    ax_traj.plot(t1.cpu().numpy(), pred_y1.cpu().numpy()[:, 0, 0], 'b--', label='pred x')
    ax_traj.plot(t1.cpu().numpy(), pred_y1.cpu().numpy()[:, 0, 1], 'g--', label='pred y')
    ax_traj.set_xlim(t1.cpu().min(), t1.cpu().max())
    ax_traj.set_ylim(-100, 100)
    ax_traj.set_title('Trajectories (left gyre)')
    ax_traj.set_xlabel('t1')
    ax_traj.set_ylabel('x1,y1')
    ax_traj.legend()
    ax_traj.grid()

    ax_traj2.cla()
    ax_traj2.plot(t2.cpu().numpy(), true_y2.cpu().numpy()[:, 0, 0], 'b', label='true x')
    ax_traj2.plot(t2.cpu().numpy(), true_y2.cpu().numpy()[:, 0, 1], 'g', label='true y')
    ax_traj2.plot(t2.cpu().numpy(), pred_y2.cpu().numpy()[:, 0, 0], 'b--', label='pred x')
    ax_traj2.plot(t2.cpu().numpy(), pred_y2.cpu().numpy()[:, 0, 1], 'g--', label='pred y')
    ax_traj2.set_xlim(t2.cpu().min(), t2.cpu().max())
    ax_traj2.set_ylim(-100, 100)
    ax_traj2.set_title('Trajectories (right gyre)')
    ax_traj2.set_xlabel('t2')
    ax_traj2.set_ylabel('x2,y2')
    ax_traj2.legend()
    ax_traj2.grid()

    ax_vecfield.cla()
    y, x = np.mgrid[-60:60:1000j, -50:50:1000j]
    dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(1000 **2, 2)).to(device)).cpu().detach().numpy()
    dydt = dydt.reshape(1000, 1000, 2)
    ax_vecfield.plot(true_y1.cpu().numpy()[:, 0, 0], true_y1.cpu().numpy()[:, 0, 1], 'b', label='true1')
    ax_vecfield.plot(pred_y1.cpu().numpy()[:, 0, 0], pred_y1.cpu().numpy()[:, 0, 1], 'b--', label='pred1')
    ax_vecfield.plot(true_y2.cpu().numpy()[:, 0, 0], true_y2.cpu().numpy()[:, 0, 1], 'g', label='true2')
    ax_vecfield.plot(pred_y2.cpu().numpy()[:, 0, 0], pred_y2.cpu().numpy()[:, 0, 1], 'g--', label='pred2')
    ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
    ax_vecfield.set_title('Learned Vector Field')
    ax_vecfield.set_xlabel('x')
    ax_vecfield.set_ylabel('y')
    ax_vecfield.set_xlim(-50, 50)
    ax_vecfield.set_ylim(-60, 60)
    # ax_vecfield.legend()

    fig.tight_layout()
    plt.savefig('Images/{:3d}'.format(itr))
    plt.draw()
    plt.pause(0.001)

def visualize_quiver(itr, t1, t2, true_y1, true_y2, pred_y1, pred_y2, odefunc, fig, ax_traj, ax_traj2, ax_vecfield, device, debug_lvl=1):
    ax_traj.cla()
    ax_traj.plot(t1.cpu().numpy(), true_y1.cpu().numpy()[:, 0, 0], 'b', label='true x')
    ax_traj.plot(t1.cpu().numpy(), true_y1.cpu().numpy()[:, 0, 1], 'g', label='true y')
    ax_traj.plot(t1.cpu().numpy(), pred_y1.cpu().numpy()[:, 0, 0], 'b--', label='pred x')
    ax_traj.plot(t1.cpu().numpy(), pred_y1.cpu().numpy()[:, 0, 1], 'g--', label='pred y')
    ax_traj.set_xlim(t1.cpu().min(), t1.cpu().max())
    ax_traj.set_ylim(-100, 100)
    ax_traj.set_title('Trajectories (left gyre)')
    ax_traj.set_xlabel('t1')
    ax_traj.set_ylabel('x1,y1')
    ax_traj.legend()
    ax_traj.grid()

    ax_traj2.cla()
    ax_traj2.plot(t2.cpu().numpy(), true_y2.cpu().numpy()[:, 0, 0], 'b', label='true x')
    ax_traj2.plot(t2.cpu().numpy(), true_y2.cpu().numpy()[:, 0, 1], 'g', label='true y')
    ax_traj2.plot(t2.cpu().numpy(), pred_y2.cpu().numpy()[:, 0, 0], 'b--', label='pred x')
    ax_traj2.plot(t2.cpu().numpy(), pred_y2.cpu().numpy()[:, 0, 1], 'g--', label='pred y')
    ax_traj2.set_xlim(t2.cpu().min(), t2.cpu().max())
    ax_traj2.set_ylim(-100, 100)
    ax_traj2.set_title('Trajectories (right gyre)')
    ax_traj2.set_xlabel('t2')
    ax_traj2.set_ylabel('x2,y2')
    ax_traj2.legend()
    ax_traj2.grid()

    ax_vecfield.cla()
    y, x = np.mgrid[-60:60:20j, -50:50:20j]
    dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(20*20, 2)).to(device)).cpu().detach().numpy()
    dydt = dydt.reshape(20, 20, 2)
    ax_vecfield.plot(true_y1.cpu().numpy()[:, 0, 0], true_y1.cpu().numpy()[:, 0, 1], 'b', label='true1')
    ax_vecfield.plot(pred_y1.cpu().numpy()[:, 0, 0], pred_y1.cpu().numpy()[:, 0, 1], 'b--', label='pred1')
    ax_vecfield.plot(true_y2.cpu().numpy()[:, 0, 0], true_y2.cpu().numpy()[:, 0, 1], 'g', label='true2')
    ax_vecfield.plot(pred_y2.cpu().numpy()[:, 0, 0], pred_y2.cpu().numpy()[:, 0, 1], 'g--', label='pred2')
    ax_vecfield.quiver(x, y, dydt[:, :, 0], dydt[:, :, 1], units ='width')
    ax_vecfield.set_title('Learned Vector Field')
    ax_vecfield.set_xlabel('x')
    ax_vecfield.set_ylabel('y')
    ax_vecfield.set_xlim(-50, 50)
    ax_vecfield.set_ylim(-60, 60)
    # ax_vecfield.legend()

    fig.tight_layout()
    plt.savefig('Images/{:03d}'.format(itr))
    plt.draw()
    plt.pause(0.001)

def visualize_vecfield(itr, t1, t2, true_y1, true_y2, pred_y1, pred_y2, odefunc, fig, ax_traj, ax_traj2, ax_vecfield, device):
    y, x = np.mgrid[-60:60:2000j, -50:50:2000j]
    grid_samples = torch.Tensor(np.stack([x, y], -1).reshape(2000 * 2000, 2)).to(device)
    dynamics = Dynamics()
    dydt = dynamics.forward(0, grid_samples).cpu().detach().numpy()
    dydt = dydt.reshape(2000, 2000, 2)
    ax_vecfield.plot(true_y1.cpu().numpy()[:, 0, 0], true_y1.cpu().numpy()[:, 0, 1], 'b', label='true1')
    ax_vecfield.plot(true_y2.cpu().numpy()[:, 0, 0], true_y2.cpu().numpy()[:, 0, 1], 'g', label='true2')
    ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
    ax_vecfield.set_xlim(-50, 50)
    ax_vecfield.set_ylim(-60, 60)
    ax_vecfield.set_title('Vector Field')
    ax_vecfield.set_xlabel('x')
    ax_vecfield.set_ylabel('y')
    # ax_vecfield.legend()

    fig.tight_layout()
    plt.savefig('Images/{:03d}'.format(itr))
    plt.draw()
    plt.pause(0.001)

def visualize_2(itr, t1, t2, true_y1, true_y2, pred_y1, pred_y2, odefunc, fig, ax_traj, ax_traj2, ax_vecfield, device):
    ax_traj.cla()
    ax_traj.plot(t1.cpu().numpy(), true_y1.cpu().numpy()[:, 0, 0], 'b', label='true x')
    ax_traj.plot(t1.cpu().numpy(), true_y1.cpu().numpy()[:, 0, 1], 'g', label='true y')
    ax_traj.plot(t1.cpu().numpy(), pred_y1.cpu().numpy()[:, 0, 0], 'b--', label='pred x')
    ax_traj.plot(t1.cpu().numpy(), pred_y1.cpu().numpy()[:, 0, 1], 'g--', label='pred y')
    ax_traj.set_xlim(t1.cpu().min(), t1.cpu().max())
    ax_traj.set_ylim(-100, 100)
    ax_traj.set_title('Extrapolated Trajectories (left gyre)')
    ax_traj.set_xlabel('t1')
    ax_traj.set_ylabel('x1,y1')
    ax_traj.legend()
    ax_traj.grid()

    ax_traj2.cla()
    ax_traj2.plot(t2.cpu().numpy(), true_y2.cpu().numpy()[:, 0, 0], 'b', label='true x')
    ax_traj2.plot(t2.cpu().numpy(), true_y2.cpu().numpy()[:, 0, 1], 'g', label='true y')
    ax_traj2.plot(t2.cpu().numpy(), pred_y2.cpu().numpy()[:, 0, 0], 'b--', label='pred x')
    ax_traj2.plot(t2.cpu().numpy(), pred_y2.cpu().numpy()[:, 0, 1], 'g--', label='pred y')
    ax_traj2.set_xlim(t2.cpu().min(), t2.cpu().max())
    ax_traj2.set_ylim(-100, 100)
    ax_traj2.set_title('Extrapolated Trajectories (right gyre)')
    ax_traj2.set_xlabel('t2')
    ax_traj2.set_ylabel('x2,y2')
    ax_traj2.legend()
    ax_traj2.grid()

    ax_vecfield.cla()
    y, x = np.mgrid[-60:60:1000j, -50:50:1000j]
    dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(1000 **2, 2)).to(device)).cpu().detach().numpy()
    dydt = dydt.reshape(1000, 1000, 2)
    ax_vecfield.plot(true_y1.cpu().numpy()[:, 0, 0], true_y1.cpu().numpy()[:, 0, 1], 'b', label='true1')
    ax_vecfield.plot(pred_y1.cpu().numpy()[:, 0, 0], pred_y1.cpu().numpy()[:, 0, 1], 'b--', label='pred1')
    ax_vecfield.plot(true_y2.cpu().numpy()[:, 0, 0], true_y2.cpu().numpy()[:, 0, 1], 'g', label='true2')
    ax_vecfield.plot(pred_y2.cpu().numpy()[:, 0, 0], pred_y2.cpu().numpy()[:, 0, 1], 'g--', label='pred2')
    ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
    ax_vecfield.set_title('Learned Vector Field')
    ax_vecfield.set_xlabel('x')
    ax_vecfield.set_ylabel('y')
    ax_vecfield.set_xlim(-50, 50)
    ax_vecfield.set_ylim(-60, 60)
    # ax_vecfield.legend()

    fig.tight_layout()
    plt.savefig('Images/Extrapolation_{:03d}'.format(itr))
    plt.draw()
    plt.pause(0.001)





# def visualize(itr, t1, t2, t3, true_y1, true_y2, true_y3, pred_y1, pred_y2, pred_y3, odefunc, fig, ax_traj, ax_traj2, ax_traj3, ax_vecfield, device):
#     ax_traj.cla()
#     ax_traj.plot(t1.cpu().numpy(), true_y1.cpu().numpy()[:, 0, 0], 'b', label='true x')
#     ax_traj.plot(t1.cpu().numpy(), true_y1.cpu().numpy()[:, 0, 1], 'g', label='true y')
#     ax_traj.plot(t1.cpu().numpy(), pred_y1.cpu().numpy()[:, 0, 0], 'b--', label='pred x')
#     ax_traj.plot(t1.cpu().numpy(), pred_y1.cpu().numpy()[:, 0, 1], 'g--', label='pred y')
#     ax_traj.set_xlim(t1.cpu().min(), t1.cpu().max())
#     ax_traj.set_ylim(-60, 10)
#     ax_traj.set_title('Trajectories (left gyre)')
#     ax_traj.set_xlabel('t1')
#     ax_traj.set_ylabel('x1,y1')
#     ax_traj.legend()
#     ax_traj.grid()
#
#     ax_traj2.cla()
#     ax_traj2.plot(t2.cpu().numpy(), true_y2.cpu().numpy()[:, 0, 0], 'b', label='true x')
#     ax_traj2.plot(t3.cpu().numpy(), true_y3.cpu().numpy()[:, 0, 0], 'b', label='true x')
#     ax_traj2.plot(t2.cpu().numpy(), true_y2.cpu().numpy()[:, 0, 1], 'g', label='true y')
#     ax_traj2.plot(t3.cpu().numpy(), true_y3.cpu().numpy()[:, 0, 1], 'g', label='true y')
#     ax_traj2.plot(t2.cpu().numpy(), pred_y2.cpu().numpy()[:, 0, 0], 'b--', label='pred x')
#     ax_traj2.plot(t3.cpu().numpy(), pred_y3.cpu().numpy()[:, 0, 0], 'b--', label='pred x')
#     ax_traj2.plot(t2.cpu().numpy(), pred_y2.cpu().numpy()[:, 0, 1], 'g--', label='pred y')
#     ax_traj2.plot(t3.cpu().numpy(), pred_y3.cpu().numpy()[:, 0, 1], 'g--', label='pred y')
#     ax_traj2.set_xlim(t2.cpu().min(), t2.cpu().max())
#     ax_traj2.set_ylim(-50, 40)
#     ax_traj2.set_title('Trajectories (right gyre)')
#     ax_traj2.set_xlabel('t2')
#     ax_traj2.set_ylabel('x2,y2')
#     ax_traj2.legend()
#     ax_traj2.grid()
#
#     ax_vecfield.cla()
#     y, x = np.mgrid[-60:10:1000j, -50:50:1000j]
#     dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(1000 **2, 2)).to(device)).cpu().detach().numpy()
#     dydt = dydt.reshape(1000, 1000, 2)
#     ax_vecfield.plot(true_y1.cpu().numpy()[:, 0, 0], true_y1.cpu().numpy()[:, 0, 1], 'b', label='true1')
#     ax_vecfield.plot(true_y3.cpu().numpy()[:, 0, 0], true_y3.cpu().numpy()[:, 0, 1], 'b', label='true1')
#     ax_vecfield.plot(pred_y1.cpu().numpy()[:, 0, 0], pred_y1.cpu().numpy()[:, 0, 1], 'b--', label='pred1')
#     ax_vecfield.plot(pred_y3.cpu().numpy()[:, 0, 0], pred_y3.cpu().numpy()[:, 0, 1], 'b--', label='pred1')
#     ax_vecfield.plot(true_y2.cpu().numpy()[:, 0, 0], true_y2.cpu().numpy()[:, 0, 1], 'g', label='true2')
#     ax_vecfield.plot(pred_y2.cpu().numpy()[:, 0, 0], pred_y2.cpu().numpy()[:, 0, 1], 'g--', label='pred2')
#     ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
#     ax_vecfield.set_title('Learned Vector Field')
#     ax_vecfield.set_xlabel('x')
#     ax_vecfield.set_ylabel('y')
#     ax_vecfield.set_xlim(-50, 50)
#     ax_vecfield.set_ylim(-60, 10)
#     # ax_vecfield.legend()
#
#     fig.tight_layout()
#     plt.savefig('Images/{:03d}'.format(itr))
#     plt.draw()
#     plt.pause(0.001)
def visualize_vector_field2(itr, odefunc, fig, ax_traj, ax_traj2, ax_vecfield, device):
    ax_vecfield.cla()
    y, x = np.mgrid[-60:60:1000j, -50:50:1000j]
    dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(1000 ** 2, 2)).to(device)).cpu().detach().numpy()
    dydt = dydt.reshape(1000, 1000, 2)
    ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
    ax_vecfield.set_title('Learned Vector Field')
    ax_vecfield.set_xlabel('x')
    ax_vecfield.set_ylabel('y')
    ax_vecfield.set_xlim(-50, 50)
    ax_vecfield.set_ylim(-60, 60)
    # ax_vecfield.legend()

    fig.tight_layout()
    plt.savefig('Images/Learned_Vector_Field_{:03d}'.format(itr))
    plt.draw()
    plt.pause(0.001)


# New visulisation functions: 4:07 am Nov 22, 2022
####################################################################################
#######################################################################################
# visualize_true_single_gyre(true_init_cond_traj_1, true_time_traj_1, device)
# visualize_true_quiver_single_gyre(true_init_cond_traj_1, true_time_traj_1, device)
def visualize_true_double_gyre(t1, true_y1, t2, true_y2, device, model_type, flow_type):
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_traj2 = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)

    ax_traj.cla()
    ax_traj.plot(t1.cpu().numpy(), true_y1.cpu().numpy()[:, 0, 0], 'b', label='true x')
    ax_traj.plot(t1.cpu().numpy(), true_y1.cpu().numpy()[:, 0, 1], 'g', label='true y')
    ax_traj.set_xlim(t1.cpu().min(), t1.cpu().max())
    ax_traj.set_ylim(-100, 100)
    ax_traj.set_title('Trajectories (left gyre)')
    ax_traj.set_xlabel('t')
    ax_traj.set_ylabel('x,y')
    ax_traj.legend()
    ax_traj.grid('on')

    ax_traj2.cla()
    ax_traj2.plot(t2.cpu().numpy(), true_y2.cpu().numpy()[:, 0, 0], 'b', label='true x')
    ax_traj2.plot(t2.cpu().numpy(), true_y2.cpu().numpy()[:, 0, 1], 'g', label='true y')
    ax_traj2.set_xlim(t2.cpu().min(), t2.cpu().max())
    ax_traj2.set_ylim(-100, 100)
    ax_traj2.set_title('Trajectories (right gyre)')
    ax_traj2.set_xlabel('t2')
    ax_traj2.set_ylabel('x2,y2')
    ax_traj2.legend()
    ax_traj2.grid()

    y, x = np.mgrid[-25:75:1000j, -50:50:1000j]
    grid_samples = torch.Tensor(np.stack([x, y], -1).reshape(1000 * 1000, 2)).to(device)
    dynamics = Dynamics()
    dydt = dynamics.forward(0, grid_samples).cpu().detach().numpy()
    dydt = dydt.reshape(1000, 1000, 2)
    ax_vecfield.plot(true_y1.cpu().numpy()[:, 0, 0], true_y1.cpu().numpy()[:, 0, 1], 'b', label='true1')
    ax_vecfield.plot(true_y2.cpu().numpy()[:, 0, 0], true_y2.cpu().numpy()[:, 0, 1], 'g', label='true2')
    ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
    ax_vecfield.set_xlim(-50, 50)
    ax_vecfield.set_ylim(-25, 75)
    ax_vecfield.set_title('Vector Field')
    ax_vecfield.set_xlabel('x')
    ax_vecfield.set_ylabel('y')

    fig.tight_layout()
    plt.draw()
    plt.savefig('Images/' + flow_type + 'Double_Gyre/'+model_type+'/true')

def visualize_true_quiver_double_gyre(t1, true_y1, t2, true_y2, device):
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_traj2 = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)

    ax_traj.cla()
    ax_traj.plot(t1.cpu().numpy(), true_y1.cpu().numpy()[:, 0, 0], 'b', label='true x')
    ax_traj.plot(t1.cpu().numpy(), true_y1.cpu().numpy()[:, 0, 1], 'g', label='true y')
    ax_traj.set_xlim(t1.cpu().min(), t1.cpu().max())
    ax_traj.set_ylim(-60, 60)
    ax_traj.set_title('Trajectories (left gyre)')
    ax_traj.set_xlabel('t')
    ax_traj.set_ylabel('x,y')
    ax_traj.legend()
    ax_traj.grid('on')

    ax_traj2.cla()
    ax_traj2.plot(t2.cpu().numpy(), true_y2.cpu().numpy()[:, 0, 0], 'b', label='true x')
    ax_traj2.plot(t2.cpu().numpy(), true_y2.cpu().numpy()[:, 0, 1], 'g', label='true y')
    ax_traj2.set_xlim(t2.cpu().min(), t2.cpu().max())
    ax_traj2.set_ylim(-50, 50)
    ax_traj2.set_title('Trajectories (right gyre)')
    ax_traj2.set_xlabel('t2')
    ax_traj2.set_ylabel('x2,y2')
    ax_traj2.legend()
    ax_traj2.grid()

    res = 100
    y, x = np.mgrid[-60:60:20j, -50:50:20j]
    grid_samples = torch.Tensor(np.stack([x, y], -1).reshape(20 * 20, 2)).to(device)
    dynamics = Dynamics()
    dydt = dynamics.forward(0, grid_samples).cpu().detach().numpy()
    dydt = dydt.reshape(20, 20, 2)
    ax_vecfield.plot(true_y1.cpu().numpy()[:, 0, 0], true_y1.cpu().numpy()[:, 0, 1], 'b', label='true1')
    ax_vecfield.plot(true_y2.cpu().numpy()[:, 0, 0], true_y2.cpu().numpy()[:, 0, 1], 'g', label='true2')
    ax_vecfield.quiver(x, y, dydt[:, :, 0], dydt[:, :, 1])
    ax_vecfield.set_xlim(-50, 50)
    ax_vecfield.set_ylim(-60, 60)
    ax_vecfield.set_title('Vector Field')
    ax_vecfield.set_xlabel('x')
    ax_vecfield.set_ylabel('y')

    fig.tight_layout()
    plt.draw()

# visualize_single_gyre_streamplot(itr, true_time_traj_1, true_traj_1, pred_traj_1 + knwlge_based_traj_1, func,
#                                  fig_s, ax_traj_s1, ax_traj_s2, ax_vecfield_s, device)
# visualize_single_gyre_quiverplot(itr, true_time_traj_1, true_traj_1, pred_traj_1 + knwlge_based_traj_1, func,
#                                  fig_q, ax_traj_q1, ax_traj_q2, ax_vecfield_q, device)

def visualize_double_gyre_streamplot(itr, t1, t2, true_y1, true_y2, pred_y1, pred_y2, odefunc, fig, ax_traj, ax_traj2, ax_vecfield, device, gyre_type, model_type, flow_type):
    ax_traj.cla()
    ax_traj.plot(t1.cpu().numpy(), true_y1.cpu().numpy()[:, 0, 0], 'b', label='true x')
    ax_traj.plot(t1.cpu().numpy(), true_y1.cpu().numpy()[:, 0, 1], 'g', label='true y')
    ax_traj.plot(t1.cpu().numpy(), pred_y1.cpu().numpy()[:, 0, 0],  color ='rebeccapurple', linestyle='dashed', label='pred x')
    ax_traj.plot(t1.cpu().numpy(), pred_y1.cpu().numpy()[:, 0, 1], color ='yellow', linestyle='dashed', label='pred y')
    ax_traj.set_xlim(t1.cpu().min(), t1.cpu().max())
    ax_traj.set_ylim(-100, 100)
    ax_traj.set_title('Trajectories (left gyre)')
    ax_traj.set_xlabel('t1')
    ax_traj.set_ylabel('x1,y1')
    ax_traj.legend()
    ax_traj.grid()

    ax_traj2.cla()
    ax_traj2.plot(t2.cpu().numpy(), true_y2.cpu().numpy()[:, 0, 0], 'b', label='true x')
    ax_traj2.plot(t2.cpu().numpy(), true_y2.cpu().numpy()[:, 0, 1], 'g', label='true y')
    ax_traj2.plot(t2.cpu().numpy(), pred_y2.cpu().numpy()[:, 0, 0], color ='rebeccapurple', linestyle='dashed', label='pred x')
    ax_traj2.plot(t2.cpu().numpy(), pred_y2.cpu().numpy()[:, 0, 1], color ='yellow', linestyle='dashed', label='pred y')
    ax_traj2.set_xlim(t2.cpu().min(), t2.cpu().max())
    ax_traj2.set_ylim(-100, 100)
    ax_traj2.set_title('Trajectories (right gyre)')
    ax_traj2.set_xlabel('t2')
    ax_traj2.set_ylabel('x2,y2')
    ax_traj2.legend()
    ax_traj2.grid()

    ax_vecfield.cla()
    y, x = np.mgrid[-25:75:1000j, -50:50:1000j]
    if flow_type == "Var_":
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(1000 ** 2, 1,2)).to(device)).cpu().detach().numpy()
    else:
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(1000 **2, 2)).to(device)).cpu().detach().numpy()
    dydt = dydt.reshape(1000, 1000, 2)
    ax_vecfield.plot(true_y1.cpu().numpy()[:, 0, 0], true_y1.cpu().numpy()[:, 0, 1], 'b', label='true1')
    ax_vecfield.plot(pred_y1.cpu().numpy()[:, 0, 0], pred_y1.cpu().numpy()[:, 0, 1], color ='rebeccapurple', linestyle='dashed', label='pred1')
    ax_vecfield.plot(true_y2.cpu().numpy()[:, 0, 0], true_y2.cpu().numpy()[:, 0, 1], 'g', label='true2')
    ax_vecfield.plot(pred_y2.cpu().numpy()[:, 0, 0], pred_y2.cpu().numpy()[:, 0, 1], color ='yellow', linestyle='dashed', label='pred2')
    ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
    ax_vecfield.set_title('Learned Vector Field:{:03d}'.format(itr))
    ax_vecfield.set_xlabel('x')
    ax_vecfield.set_ylabel('y')
    ax_vecfield.set_xlim(-50, 50)
    ax_vecfield.set_ylim(-25, 75)
    # ax_vecfield.legend()

    fig.tight_layout()
    plt.draw()

    fig.savefig('Images/'+ flow_type + 'Double_Gyre/'+model_type +'/pred_{:2d}'.format(itr))
    plt.pause(0.001)

def visualize_double_gyre_quiverplot(itr, t1, t2, true_y1, true_y2, pred_y1, pred_y2, odefunc, fig, ax_traj, ax_traj2, ax_vecfield,cbar_ax, device):

    ax_traj.cla()
    ax_traj.plot(t1.cpu().numpy(), true_y1.cpu().numpy()[:, 0, 0], 'b', label='true x')
    ax_traj.plot(t1.cpu().numpy(), true_y1.cpu().numpy()[:, 0, 1], 'g', label='true y')
    ax_traj.plot(t1.cpu().numpy(), pred_y1.cpu().numpy()[:, 0, 0], 'b--', label='pred x')
    ax_traj.plot(t1.cpu().numpy(), pred_y1.cpu().numpy()[:, 0, 1], 'g--', label='pred y')
    ax_traj.set_xlim(t1.cpu().min(), t1.cpu().max())
    ax_traj.set_ylim(-100, 100)
    ax_traj.set_title('Trajectories (left gyre)')
    ax_traj.set_xlabel('t1')
    ax_traj.set_ylabel('x1,y1')
    ax_traj.legend()
    ax_traj.grid()

    ax_traj2.cla()
    ax_traj2.plot(t2.cpu().numpy(), true_y2.cpu().numpy()[:, 0, 0], 'b', label='true x')
    ax_traj2.plot(t2.cpu().numpy(), true_y2.cpu().numpy()[:, 0, 1], 'g', label='true y')
    ax_traj2.plot(t2.cpu().numpy(), pred_y2.cpu().numpy()[:, 0, 0], 'b--', label='pred x')
    ax_traj2.plot(t2.cpu().numpy(), pred_y2.cpu().numpy()[:, 0, 1], 'g--', label='pred y')
    ax_traj2.set_xlim(t2.cpu().min(), t2.cpu().max())
    ax_traj2.set_ylim(-100, 100)
    ax_traj2.set_title('Trajectories (right gyre)')
    ax_traj2.set_xlabel('t2')
    ax_traj2.set_ylabel('x2,y2')
    ax_traj2.legend()
    ax_traj2.grid()

    ax_vecfield.cla()
    cbar_ax.cla()
    y, x = np.mgrid[-60:60:20j, -50:50:20j]
    dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(20*20, 2)).to(device)).cpu().detach().numpy()
    dydt = dydt.reshape(20, 20, 2)
    M    = np.sqrt(dydt[:, :, 0] * dydt[:, :, 0] + dydt[:, :, 1] * dydt[:, :, 1])
    ax_vecfield.plot(true_y1.cpu().numpy()[:, 0, 0], true_y1.cpu().numpy()[:, 0, 1], 'b', label='true1')
    ax_vecfield.plot(pred_y1.cpu().numpy()[:, 0, 0], pred_y1.cpu().numpy()[:, 0, 1], 'b--', label='pred1')
    ax_vecfield.plot(true_y2.cpu().numpy()[:, 0, 0], true_y2.cpu().numpy()[:, 0, 1], 'g', label='true2')
    ax_vecfield.plot(pred_y2.cpu().numpy()[:, 0, 0], pred_y2.cpu().numpy()[:, 0, 1], 'g--', label='pred2')
    qq = ax_vecfield.quiver(x, y, dydt[:, :, 0], dydt[:, :, 1], M,units ='width',cmap=plt.cm.jet)
    ax_vecfield.set_title('Learned Vector Field')
    ax_vecfield.set_xlabel('x')
    ax_vecfield.set_ylabel('y')
    ax_vecfield.set_xlim(-50, 50)
    ax_vecfield.set_ylim(-60, 60)
    cbar = plt.colorbar(qq, cax =cbar_ax,cmap=plt.cm.jet)

    # ax_vecfield.legend()

    fig.tight_layout()
    plt.savefig('Images/{:3d}'.format(itr))
    plt.draw()
    plt.pause(0.001)
    return cbar

def visualize_single_gyre_streamplot(itr, t1,true_y1, pred_y1, odefunc, fig, ax_traj, ax_vecfield, device, gyre_type, model_type, flow_type):
    ax_traj.cla()
    ax_traj.plot(t1.cpu().numpy(), true_y1.cpu().numpy()[:, 0, 0], 'b', label='true x')
    ax_traj.plot(t1.cpu().numpy(), true_y1.cpu().numpy()[:, 0, 1], 'g', label='true y')
    ax_traj.plot(t1.cpu().numpy(), pred_y1.cpu().numpy()[:, 0, 0],  color ='rebeccapurple', linestyle='dashed', label='pred x')
    ax_traj.plot(t1.cpu().numpy(), pred_y1.cpu().numpy()[:, 0, 1], color ='yellow', linestyle='dashed', label='pred y')
    ax_traj.set_xlim(t1.cpu().min(), t1.cpu().max())
    ax_traj.set_ylim(-100, 100)
    ax_traj.set_title('Trajectories (left gyre)')
    ax_traj.set_xlabel('t1')
    ax_traj.set_ylabel('x1,y1')
    ax_traj.legend()
    ax_traj.grid()

    ax_vecfield.cla()
    y, x = np.mgrid[0:50:1000j, -50:0:1000j]
    if flow_type == 'Var_':
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(1000 ** 2,1, 2)).to(device)).cpu().detach().numpy()
    else:
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(1000 **2, 2)).to(device)).cpu().detach().numpy()
    dydt = dydt.reshape(1000, 1000, 2)
    ax_vecfield.plot(true_y1.cpu().numpy()[:, 0, 0], true_y1.cpu().numpy()[:, 0, 1], color='orange', label='ground_truth')
    ax_vecfield.plot(pred_y1.cpu().numpy()[:, 0, 0], pred_y1.cpu().numpy()[:, 0, 1], color ='red', linestyle='dashed', label='predicted')
    # ax_vecfield.plot(true_y2.cpu().numpy()[:, 0, 0], true_y2.cpu().numpy()[:, 0, 1], 'g', label='true2')
    # ax_vecfield.plot(pred_y2.cpu().numpy()[:, 0, 0], pred_y2.cpu().numpy()[:, 0, 1], color ='yellow', linestyle='dashed', label='pred2')
    ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
    ax_vecfield.set_title('Learned Vector Field:{:03d}'.format(itr))
    ax_vecfield.set_xlabel('x')
    ax_vecfield.set_ylabel('y')
    ax_vecfield.set_xlim(-50,0)
    ax_vecfield.set_ylim(0, 50)
    ax_vecfield.legend()

    fig.tight_layout()
    plt.draw()
    fig.savefig('Images/' + flow_type + 'Single_Gyre/' + model_type+'/pred_{:2d}'.format(itr))
    plt.pause(0.001)

def visualize_true_single_gyre(t1, true_y1, device, model_type, flow_type):
    fig = plt.figure(figsize=(8, 4), facecolor='white')
    ax_traj = fig.add_subplot(121, frameon=False)
    ax_vecfield = fig.add_subplot(122, frameon=False)

    ax_traj.cla()
    ax_traj.plot(t1.cpu().numpy(), true_y1.cpu().numpy()[:, 0, 0], 'b', label='true x')
    ax_traj.plot(t1.cpu().numpy(), true_y1.cpu().numpy()[:, 0, 1], 'g', label='true y')
    ax_traj.set_xlim(t1.cpu().min(), t1.cpu().max())
    ax_traj.set_ylim(-60, 60)
    ax_traj.set_title('Trajectories (left gyre)')
    ax_traj.set_xlabel('t')
    ax_traj.set_ylabel('x,y')
    ax_traj.legend()
    ax_traj.grid('on')

    y, x = np.mgrid[0:50:1000j, -50:0:1000j]
    grid_samples = torch.Tensor(np.stack([x, y], -1).reshape(1000 * 1000, 2)).to(device)
    dynamics = Dynamics()
    dydt = dynamics.forward(0, grid_samples).cpu().detach().numpy()
    dydt = dydt.reshape(1000, 1000, 2)
    ax_vecfield.plot(true_y1.cpu().numpy()[:, 0, 0], true_y1.cpu().numpy()[:, 0, 1], 'orange', label='ground_truth')
    ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
    ax_vecfield.set_xlim(-50,0)
    ax_vecfield.set_ylim(0, 50)
    ax_vecfield.set_title('Vector Field')
    ax_vecfield.set_xlabel('x')
    ax_vecfield.set_ylabel('y')

    fig.tight_layout()
    plt.draw()
    plt.savefig('Images/'+ flow_type + 'Single_Gyre/'+model_type+'/true')


def  visualize_err_vecfield(itr, true,func, fig_q, ax_true_vecfield, ax_pred_vecfield , ax_err_vecfield, cbar_ax_1, cbar_ax_2, cbar_ax_3, device,gyre_type, model_type, flow_type):
    ax_true_vecfield.cla()
    cbar_ax_1.cla()
    if gyre_type == "single":
        y, x = np.mgrid[0:50:1000j, -50:0:1000j]
    else:
        y, x = np.mgrid[-25:75:1000j, -50:50:1000j]

    grid_samples = torch.Tensor(np.stack([x, y], -1).reshape(1000 * 1000, 2)).to(device)
    dynamics = true
    dydt = dynamics.forward(0, grid_samples).cpu().detach().numpy()
    dydt = dydt.reshape(1000, 1000, 2)
    M = np.sqrt(dydt[:, :, 0]**2 + dydt[:, :, 1]**2)
    ax_true_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
    if gyre_type == "single":
        ax_true_vecfield.set_xlim(-50, 0)
        ax_true_vecfield.set_ylim(0, 50)
    else:
        ax_true_vecfield.set_xlim(-50, 50)
        ax_true_vecfield.set_ylim(-25, 75)
    ax_true_vecfield.set_title('True Vector Field')
    ax_true_vecfield.set_xlabel('x')
    ax_true_vecfield.set_ylabel('y')
    q1 = ax_true_vecfield.contourf(x, y, M, cmap=plt.cm.cividis)
    plt.colorbar(q1, cax=cbar_ax_1, cmap=plt.cm.cividis)

    ax_pred_vecfield.cla()
    cbar_ax_2.cla()
    if flow_type == "Var_":
        dydt_nn = func.forward(0, grid_samples.unsqueeze(dim=1)).cpu().detach().numpy()
    else:
        dydt_nn = func.forward(0, grid_samples).cpu().detach().numpy()
    dydt_nn = dydt_nn.reshape(1000, 1000, 2)
    M_nn = np.sqrt(dydt_nn[:, :, 0]**2 + dydt_nn[:, :, 1]**2)
    ax_pred_vecfield.streamplot(x, y, dydt_nn[:, :, 0], dydt_nn[:, :, 1], color="black")
    q2 = ax_pred_vecfield.contourf(x, y, M_nn, cmap=plt.cm.cividis)
    if gyre_type == "single":
        ax_pred_vecfield.set_xlim(-50, 0)
        ax_pred_vecfield.set_ylim(0, 50)
    else:
        ax_pred_vecfield.set_xlim(-50, 50)
        ax_pred_vecfield.set_ylim(-25, 75)
    ax_pred_vecfield.set_title('Pred Vector Field:{:03d}'.format(itr))
    ax_pred_vecfield.set_xlabel('x')
    ax_pred_vecfield.set_ylabel('y')
    plt.colorbar(q2, cax=cbar_ax_2, cmap=plt.cm.cividis)

    ax_err_vecfield.cla()
    cbar_ax_3.cla()
    dydt_err = dydt -dydt_nn
    M_err    = np.sqrt(dydt_err[:, :, 0]**2 + dydt_err[:, :, 1]**2)
    q3 = ax_err_vecfield.contourf(x, y, M_err, cmap=plt.cm.cividis)
    if gyre_type == "single":
        ax_err_vecfield.set_xlim(-50, 0)
        ax_err_vecfield.set_ylim(0, 50)
    else:
        ax_err_vecfield.set_xlim(-50, 50)
        ax_err_vecfield.set_ylim(-25, 75)
    ax_err_vecfield.set_title('Error:{:03d}'.format(itr))
    ax_err_vecfield.set_xlabel('x')
    ax_err_vecfield.set_ylabel('y')
    plt.colorbar(q3, cax=cbar_ax_3, cmap=plt.cm.cividis)

    fig_q.tight_layout()
    plt.draw()
    if gyre_type == "single":
        plt.savefig('Images/'+ flow_type +'Single_Gyre/'+model_type+'/vector_fields_{:2d}'.format(itr))
    else:
        plt.savefig('Images/'+ flow_type +'Double_Gyre/'+model_type+'/vector_fields_{:2d}'.format(itr))
    plt.pause(0.001)

def  visualize_err_vecfield_knode(itr, true,func, fig_q, ax_true_vecfield, ax_pred_vecfield , cbar_ax_1, cbar_ax_2,  device,gyre_type, model_type):
    ax_true_vecfield.cla()
    cbar_ax_1.cla()
    if gyre_type == "single":
        y, x = np.mgrid[0:50:1000j, -50:0:1000j]
    else:
        y, x = np.mgrid[-25:75:1000j, -50:50:1000j]

    grid_samples = torch.Tensor(np.stack([x, y], -1).reshape(1000 * 1000, 2)).to(device)
    dynamics = true
    dydt = dynamics.forward(0, grid_samples).cpu().detach().numpy()
    dydt = dydt.reshape(1000, 1000, 2)
    M = np.sqrt(dydt[:, :, 0]**2 + dydt[:, :, 1]**2)
    ax_true_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
    if gyre_type == "single":
        ax_true_vecfield.set_xlim(-50, 0)
        ax_true_vecfield.set_ylim(0, 50)
    else:
        ax_true_vecfield.set_xlim(-50, 50)
        ax_true_vecfield.set_ylim(-25, 75)
    ax_true_vecfield.set_title('True Vector Field')
    ax_true_vecfield.set_xlabel('x')
    ax_true_vecfield.set_ylabel('y')
    q1 = ax_true_vecfield.contourf(x, y, M, cmap=plt.cm.cividis)
    plt.colorbar(q1, cax=cbar_ax_1, cmap=plt.cm.cividis)

    ax_pred_vecfield.cla()
    cbar_ax_2.cla()
    dydt_nn = func.forward(0, grid_samples).cpu().detach().numpy()
    dydt_nn = dydt_nn.reshape(1000, 1000, 2)
    M_nn = np.sqrt(dydt_nn[:, :, 0]**2 + dydt_nn[:, :, 1]**2)
    ax_pred_vecfield.streamplot(x, y, dydt_nn[:, :, 0], dydt_nn[:, :, 1], color="black")
    q2 = ax_pred_vecfield.contourf(x, y, M_nn, cmap=plt.cm.cividis)
    if gyre_type == "single":
        ax_pred_vecfield.set_xlim(-50, 0)
        ax_pred_vecfield.set_ylim(0, 50)
    else:
        ax_pred_vecfield.set_xlim(-50, 50)
        ax_pred_vecfield.set_ylim(-25, 75)
    ax_pred_vecfield.set_title('NN vector field:{:03d}'.format(itr))
    ax_pred_vecfield.set_xlabel('x')
    ax_pred_vecfield.set_ylabel('y')
    plt.colorbar(q2, cax=cbar_ax_2, cmap=plt.cm.cividis)

    fig_q.tight_layout()
    plt.draw()
    if gyre_type == "single":
        fig_q.savefig('Images/Inv_Single_Gyre/' + model_type+ '/vector_fields_{:2d}'.format(itr))
    else:
        plt.savefig('Images/Inv_Double_Gyre/' + model_type+'/vector_fields_{:2d}'.format(itr))
    plt.pause(0.001)

