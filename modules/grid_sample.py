import torch


def grid_sample(x, grid):
    x_h, x_w = x.shape[2], x.shape[3]
    i, j = grid[..., 1], grid[..., 0]
    i = ((x_h - 1) * (i + 1) / 2).view(x.shape[0], -1)
    j = ((x_w - 1) * (j + 1) / 2).view(x.shape[0], -1)

    i_1 = torch.clamp_max(torch.clamp_min(torch.floor(i), 0), x_h - 1).long()
    i_2 = torch.clamp_min(torch.clamp_max(i_1 + 1, x_h - 1), 0).long()
    j_1 = torch.clamp_max(torch.clamp_min(torch.floor(j), 0), x_w - 1).long()
    j_2 = torch.clamp_min(torch.clamp_max(j_1 + 1, x_w - 1), 0).long()

    v = x.view(x.shape[0], x.shape[1], -1)
    q_11 = torch.gather(v, dim=-1, index=(i_1 * x_w + j_1).long().unsqueeze(dim=1).expand(v.shape[0], v.shape[1], v.shape[2]))
    q_12 = torch.gather(v, dim=-1, index=(i_1 * x_w + j_2).long().unsqueeze(dim=1).expand(v.shape[0], v.shape[1], v.shape[2]))
    q_21 = torch.gather(v, dim=-1, index=(i_2 * x_w + j_1).long().unsqueeze(dim=1).expand(v.shape[0], v.shape[1], v.shape[2]))
    q_22 = torch.gather(v, dim=-1, index=(i_2 * x_w + j_2).long().unsqueeze(dim=1).expand(v.shape[0], v.shape[1], v.shape[2]))

    di = (i - i_1).unsqueeze(dim=1)
    dj = (j - j_1).unsqueeze(dim=1)

    q_i1 = q_11 * (1 - di) + q_21 * di
    q_i2 = q_12 * (1 - di) + q_22 * di
    q_ij = q_i1 * (1 - dj) + q_i2 * dj
    return q_ij.view_as(x)
