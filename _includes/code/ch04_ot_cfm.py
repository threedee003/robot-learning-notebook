import torch
from scipy.optimize import linear_sum_assignment

from ch02_ddpm_2d import eight_gaussians
from ch04_cfm_2d import train_cfm, report


def minibatch_ot(x0, x1):
      cost = torch.cdist(x0, x1) ** 2
      rows, cols = linear_sum_assignment(cost.numpy())
      return x0[rows], x1[cols]


@torch.no_grad()
def reflow_pairs(model, fm, n, steps=50):
      x0 = torch.randn(n, 2)
      x1 = fm.sample(model, (n, 2), steps=steps, x0=x0)
      return x0, x1


if __name__ == "__main__":
      _, centers = eight_gaussians(1)
      model_ind, fm = train_cfm()
      report("independent", model_ind, fm, centers)
      model_ot, fm = train_cfm(coupling=minibatch_ot)
      report("minibatch-OT", model_ot, fm, centers)
