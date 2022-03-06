import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad


class FlowGradientReg(nn.Module):
    def __init__(self):
        super(FlowGradientReg, self).__init__()
        self.flow_matrx = None

    def forward(self, x):
        theta = torch.zeros((x.shape[0], 2, 3)).to(x.device)
        theta[:, 0, 0], theta[:, 1, 1] = theta[:, 0, 0] + 1, theta[:, 1, 1] + 1
        grid = F.affine_grid(theta, x.shape, align_corners=True)
        self.flow_matrx = Variable(torch.zeros_like(grid), requires_grad=True)
        z = self.grid_sample(x, grid + self.flow_matrx)
        return z

    def calculate(self, loss):
        gd, = grad(loss, self.flow_matrx, retain_graph=True, create_graph=True)
        reg = gd.pow(2).sum(dim=(1, 2, 3)).mean()
        return reg

    def random_start(self, x, sigma):
        theta = torch.zeros((x.shape[0], 2, 3)).to(x.device)
        theta[:, 0, 0], theta[:, 1, 1] = theta[:, 0, 0] + 1, theta[:, 1, 1] + 1
        grid = F.affine_grid(theta, x.shape, align_corners=True)
        n = torch.normal(mean=0.0, std=sigma, size=grid.shape).to(x.device)
        z = F.grid_sample(x, grid + n, align_corners=True).clamp(0, 1)
        return z.detach()

    def grid_sample(self, x, grid):
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
