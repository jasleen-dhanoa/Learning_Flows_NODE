import torch

def get_batch(t, true_y, data_size, num_traj, batch_time, device):
    s           = torch.arange(data_size*num_traj - 1)
    batch_y0    = true_y[s]
    batch_t     = t[:batch_time]
    batch_y     = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)