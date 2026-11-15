import math
import torch
import torch.nn as nn


def cosine_beta_schedule(num_steps, s=0.008):
      steps = torch.arange(num_steps + 1, dtype=torch.float64) / num_steps
      f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
      alpha_bar = f / f[0]
      betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
      return betas.clamp(max=0.999).float()


class SinusoidalEmbedding(nn.Module):
      def __init__(self, dim):
            super().__init__()
            self.dim = dim

      def forward(self, t):
            half = self.dim // 2
            freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / (half - 1))
            args = t.float()[:, None] * freqs[None]
            return torch.cat([args.sin(), args.cos()], dim=-1)


class NoiseSchedule:
      def __init__(self, num_steps=100):
            self.num_steps = num_steps
            self.betas = cosine_beta_schedule(num_steps)
            self.alphas = 1.0 - self.betas
            self.alpha_bar = torch.cumprod(self.alphas, dim=0)

      def add_noise(self, x0, noise, t):
            ab = self.alpha_bar.to(x0.device)[t].view(-1, *([1] * (x0.dim() - 1)))
            return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

      def training_loss(self, model, x0, cond=None):
            t = torch.randint(0, self.num_steps, (x0.shape[0],), device=x0.device)
            noise = torch.randn_like(x0)
            xt = self.add_noise(x0, noise, t)
            pred = model(xt, t, cond)
            return ((pred - noise) ** 2).mean()

      @torch.no_grad()
      def ddpm_sample(self, model, shape, cond=None, device="cpu", return_traj=False):
            x = torch.randn(shape, device=device)
            traj = [x.clone()]
            for t in reversed(range(self.num_steps)):
                  tt = torch.full((shape[0],), t, device=device, dtype=torch.long)
                  eps = model(x, tt, cond)
                  a, ab, b = self.alphas[t], self.alpha_bar[t], self.betas[t]
                  mean = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
                  if t > 0:
                        ab_prev = self.alpha_bar[t - 1]
                        var = b * (1 - ab_prev) / (1 - ab)
                        x = mean + var.sqrt() * torch.randn_like(x)
                  else:
                        x = mean
                  traj.append(x.clone())
            return (x, traj) if return_traj else x

      @torch.no_grad()
      def ddim_sample(self, model, shape, cond=None, num_inference_steps=10, device="cpu", clip=None):
            x = torch.randn(shape, device=device)
            ts = torch.linspace(self.num_steps - 1, 0, num_inference_steps).round().long().tolist()
            for i, t in enumerate(ts):
                  tt = torch.full((shape[0],), t, device=device, dtype=torch.long)
                  eps = model(x, tt, cond)
                  ab = self.alpha_bar[t]
                  x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
                  if clip is not None:
                        x0 = x0.clamp(-clip, clip)
                        eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
                  ab_next = self.alpha_bar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
                  x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
            return x
