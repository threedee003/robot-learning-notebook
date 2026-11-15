import math
import torch
import torch.nn as nn

from diffusion_core import NoiseSchedule, SinusoidalEmbedding


def eight_gaussians(n, radius=2.0, std=0.1):
      centers = torch.tensor([[math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)] for k in range(8)]) * radius
      idx = torch.randint(0, 8, (n,))
      return centers[idx] + std * torch.randn(n, 2), centers


class Denoiser(nn.Module):
      def __init__(self, x_dim=2, cond_dim=0, hidden=256, t_dim=64):
            super().__init__()
            self.t_embed = nn.Sequential(SinusoidalEmbedding(t_dim), nn.Linear(t_dim, t_dim), nn.SiLU())
            self.c_embed = None
            if cond_dim > 0:
                  self.c_embed = nn.Sequential(nn.Linear(cond_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
            in_dim = x_dim + t_dim + (t_dim if cond_dim > 0 else 0)
            self.net = nn.Sequential(
                  nn.Linear(in_dim, hidden), nn.SiLU(),
                  nn.Linear(hidden, hidden), nn.SiLU(),
                  nn.Linear(hidden, hidden), nn.SiLU(),
                  nn.Linear(hidden, x_dim),
            )

      def forward(self, x, t, cond=None):
            h = [x, self.t_embed(t)]
            if cond is not None:
                  h.append(self.c_embed(cond))
            return self.net(torch.cat(h, dim=-1))


def mode_stats(samples, centers, tol=0.4):
      d = torch.cdist(samples, centers)
      nearest = d.min(dim=1)
      on_mode = nearest.values < tol
      covered = torch.unique(nearest.indices[on_mode]).numel()
      return on_mode.float().mean().item(), covered


if __name__ == "__main__":
      torch.manual_seed(0)
      schedule = NoiseSchedule(num_steps=100)
      model = Denoiser()
      opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
      for it in range(4000):
            x0, centers = eight_gaussians(512)
            loss = schedule.training_loss(model, x0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if it % 1000 == 0:
                  print(f"iter {it:5d}  loss {loss.item():.4f}")

      model.eval()
      ddpm = schedule.ddpm_sample(model, (2000, 2))
      print("DDPM 100 steps  on-mode %.3f  modes covered %d/8" % mode_stats(ddpm, centers))
      for steps in (5, 10, 25, 50):
            raw = schedule.ddim_sample(model, (2000, 2), num_inference_steps=steps)
            clipped = schedule.ddim_sample(model, (2000, 2), num_inference_steps=steps, clip=3.0)
            print("DDIM %3d steps  on-mode  no clip %.3f   clip %.3f" % (
                  steps, mode_stats(raw, centers)[0], mode_stats(clipped, centers)[0]))
      torch.save(model.state_dict(), "ddpm_2d.pt")
