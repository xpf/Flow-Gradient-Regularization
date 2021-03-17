import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
from modules.grid_sample import grid_sample

class FlowGradient(nn.Module):
    def __init__(self, lambd=1.0):
        super(FlowGradient, self).__init__()
        self.flow = None
        self.lambd = lambd

    def forward(self, x):
        theta = torch.zeros((x.shape[0], 2, 3)).to(x.device)
        theta[:, 0, 0], theta[:, 1, 1] = theta[:, 0, 0] + 1, theta[:, 1, 1] + 1
        grid = F.affine_grid(theta, x.shape, align_corners=True)
        self.flow = Variable(torch.zeros_like(grid), requires_grad=True)
        z = grid_sample(x, grid + self.flow)
        return z

    def calculate(self, loss):
        gd, = grad(loss, self.flow, retain_graph=True, create_graph=True)
        reg = gd.pow(2).sum(dim=(1, 2, 3)).mean()
        return reg * self.lambd

def random_start(x, sigma):
    assert x is not None
    theta = torch.zeros((x.shape[0], 2, 3)).to(x.device)
    theta[:, 0, 0], theta[:, 1, 1] = theta[:, 0, 0] + 1, theta[:, 1, 1] + 1
    grid = F.affine_grid(theta, x.shape, align_corners=True)
    n = torch.normal(mean=0.0, std=sigma, size=grid.shape).to(x.device)
    z = F.grid_sample(x, grid + n, align_corners=True).clamp(0, 1)
    return z.data