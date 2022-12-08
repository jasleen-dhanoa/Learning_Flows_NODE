import torch

def get_batch(t, true_y, traj_lengths,data_size, num_traj, batch_time, device):
    if traj_lengths[0]==data_size*num_traj:
        s           = torch.arange(data_size*num_traj - 1)
        batch_y0    = true_y[s]
        batch_t     = t[:batch_time]
        batch_y     = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)
        return batch_y0.to(device), batch_t.to(device), batch_y.to(device)
    else:
        s_1          = torch.arange(traj_lengths[0] - 1)
        s_2          = torch.arange(traj_lengths[0] ,traj_lengths[0]+traj_lengths[1] - 1)
        batch_y0_1   = true_y[s_1]
        batch_y0_2   = true_y[s_2]
        batch_y0 = torch.cat([batch_y0_1, batch_y0_2])
        batch_t     = t[:batch_time]
        batch_y_1     = torch.stack([true_y[s_1 + i] for i in range(batch_time)], dim=0)
        batch_y_2     = torch.stack([true_y[s_2 + i] for i in range(batch_time)], dim=0)
        batch_y = torch.cat([batch_y_1, batch_y_2],dim=1)
        return batch_y0.to(device), batch_t.to(device), batch_y.to(device)
