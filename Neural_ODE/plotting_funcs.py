import matplotlib.pyplot as plt
import numpy as np
import torch
from dynamics import Dynamics

def visualize_true(t1, true_y1, t2, true_y2, device):
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_traj2 = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)

    ax_traj.cla()
    ax_traj.plot(t1.cpu().numpy(), true_y1.cpu().numpy()[:, 0, 0], 'b', label='true x')
    ax_traj.plot(t1.cpu().numpy(), true_y1.cpu().numpy()[:, 0, 1], 'g', label='true y')
    ax_traj.set_xlim(t1.cpu().min(), t1.cpu().max())
    ax_traj.set_ylim(-60, 10)
    ax_traj.set_title('Trajectories (left gyre)')
    ax_traj.set_xlabel('t')
    ax_traj.set_ylabel('x,y')
    ax_traj.legend()
    ax_traj.grid('on')

    ax_traj2.cla()
    ax_traj2.plot(t2.cpu().numpy(), true_y2.cpu().numpy()[:, 0, 0], 'b', label='true x')
    ax_traj2.plot(t2.cpu().numpy(), true_y2.cpu().numpy()[:, 0, 1], 'g', label='true y')
    ax_traj2.set_xlim(t2.cpu().min(), t2.cpu().max())
    ax_traj2.set_ylim(-50, 40)
    ax_traj2.set_title('Trajectories (right gyre)')
    ax_traj2.set_xlabel('t2')
    ax_traj2.set_ylabel('x2,y2')
    ax_traj2.legend()
    ax_traj2.grid()

    y, x = np.mgrid[-60:10:2000j, -50:50:2000j]
    grid_samples = torch.Tensor(np.stack([x, y], -1).reshape(2000 * 2000, 2)).to(device)
    dynamics = Dynamics()
    dydt = dynamics.forward(0, grid_samples).cpu().detach().numpy()
    dydt = dydt.reshape(2000, 2000, 2)
    ax_vecfield.plot(true_y1.cpu().numpy()[:, 0, 0], true_y1.cpu().numpy()[:, 0, 1], 'b', label='true1')
    ax_vecfield.plot(true_y2.cpu().numpy()[:, 0, 0], true_y2.cpu().numpy()[:, 0, 1], 'g', label='true2')
    ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
    ax_vecfield.set_xlim(-50, 50)
    ax_vecfield.set_ylim(-60, 10)
    ax_vecfield.set_title('Vector Field')
    ax_vecfield.set_xlabel('x')
    ax_vecfield.set_ylabel('y')

    fig.tight_layout()
    plt.draw()

def create_fig():
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_traj2 = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)

    return fig, ax_traj, ax_traj2, ax_vecfield


def visualize(itr, t1, t2, true_y1, true_y2, pred_y1, pred_y2, odefunc, fig, ax_traj, ax_traj2, ax_vecfield, device):
    ax_traj.cla()
    ax_traj.plot(t1.cpu().numpy(), true_y1.cpu().numpy()[:, 0, 0], 'b', label='true x')
    ax_traj.plot(t1.cpu().numpy(), true_y1.cpu().numpy()[:, 0, 1], 'g', label='true y')
    ax_traj.plot(t1.cpu().numpy(), pred_y1.cpu().numpy()[:, 0, 0], 'b--', label='pred x')
    ax_traj.plot(t1.cpu().numpy(), pred_y1.cpu().numpy()[:, 0, 1], 'g--', label='pred y')
    ax_traj.set_xlim(t1.cpu().min(), t1.cpu().max())
    ax_traj.set_ylim(-60, 10)
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
    ax_traj2.set_ylim(-50, 40)
    ax_traj2.set_title('Trajectories (right gyre)')
    ax_traj2.set_xlabel('t2')
    ax_traj2.set_ylabel('x2,y2')
    ax_traj2.legend()
    ax_traj2.grid()

    ax_vecfield.cla()
    y, x = np.mgrid[-60:10:1000j, -50:50:1000j]
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
    ax_vecfield.set_ylim(-60, 10)
    # ax_vecfield.legend()

    fig.tight_layout()
    plt.savefig('Images/{:03d}'.format(itr))
    plt.draw()
    plt.pause(0.001)