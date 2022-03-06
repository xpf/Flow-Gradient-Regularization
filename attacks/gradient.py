import torch
from torch.autograd import Variable
import torch.nn.functional as F


def multi_step(model, x, y, epsilon=0.01, iters=7):
    theta = torch.zeros((x.shape[0], 2, 3)).to(x.device)
    theta[:, 0, 0], theta[:, 1, 1] = theta[:, 0, 0] + 1, theta[:, 1, 1] + 1
    grid = F.affine_grid(theta, x.shape, align_corners=True)
    n = torch.zeros_like(grid)
    for _ in range(iters):
        n = Variable(n.data, requires_grad=True)
        m = model(F.grid_sample(x, grid + n, align_corners=True).clamp(0, 1))
        loss = -F.cross_entropy(m, y)
        model.zero_grad()
        loss.backward()
        n = n.data - n.grad.data.sign_() * epsilon / iters
        n = n.clamp(-epsilon, epsilon)
    z = F.grid_sample(x, grid + n, align_corners=True).clamp(0, 1)
    return z.data
