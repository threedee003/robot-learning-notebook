import torch

from flow_core import FlowMatcher
from ch02_ddpm_2d import Denoiser, eight_gaussians, mode_stats


def train_cfm(coupling=None, iters=4000, seed=0):
      torch.manual_seed(seed)
      fm = FlowMatcher()
      model = Denoiser()
      opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
      for it in range(iters):
            x1, _ = eight_gaussians(512)
            x0 = torch.randn_like(x1)
            if coupling is not None:
                  x0, x1 = coupling(x0, x1)
            loss = fm.training_loss(model, x1, x0=x0)
            opt.zero_grad()
            loss.backward()
            opt.step()
      return model.eval(), fm


def straightness(traj):
      steps = torch.stack(traj)
      path_len = (steps[1:] - steps[:-1]).norm(dim=-1).sum(0)
      displacement = (steps[-1] - steps[0]).norm(dim=-1)
      return (path_len.mean() / displacement.mean()).item()


def report(name, model, fm, centers):
      torch.manual_seed(123)
      x0 = torch.randn(2000, 2)
      for steps in (1, 2, 4, 10, 50):
            x = fm.sample(model, (2000, 2), steps=steps, x0=x0)
            frac, cov = mode_stats(x, centers)
            print(f"{name:12s} euler {steps:2d} steps  on-mode {frac:.3f}  modes {cov}/8")
      _, traj = fm.sample(model, (2000, 2), steps=100, x0=x0, return_traj=True)
      print(f"{name:12s} path length / displacement = {straightness(traj):.3f}")


if __name__ == "__main__":
      _, centers = eight_gaussians(1)
      model, fm = train_cfm()
      report("independent", model, fm, centers)
      torch.save(model.state_dict(), "cfm_2d.pt")
