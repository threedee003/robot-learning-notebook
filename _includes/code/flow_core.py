import torch


class FlowMatcher:
      def __init__(self, time_sampling="uniform", t_scale=1000.0, beta_s=0.999):
            self.time_sampling = time_sampling
            self.t_scale = t_scale
            self.beta = torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
            self.beta_s = beta_s

      def sample_t(self, n, device="cpu"):
            if self.time_sampling == "beta":
                  u = self.beta.sample((n,)).to(device)
                  return self.beta_s * (1.0 - u)
            return torch.rand(n, device=device)

      def velocity(self, model, x, t, cond=None):
            return model(x, t * self.t_scale, cond)

      def training_loss(self, model, x1, cond=None, x0=None):
            if x0 is None:
                  x0 = torch.randn_like(x1)
            t = self.sample_t(x1.shape[0], x1.device)
            tb = t.view(-1, *([1] * (x1.dim() - 1)))
            xt = (1 - tb) * x0 + tb * x1
            target = x1 - x0
            pred = self.velocity(model, xt, t, cond)
            return ((pred - target) ** 2).mean()

      @torch.no_grad()
      def sample(self, model, shape, cond=None, steps=10, method="euler", device="cpu", x0=None, return_traj=False):
            x = torch.randn(shape, device=device) if x0 is None else x0.clone()
            traj = [x.clone()]
            dt = 1.0 / steps
            for i in range(steps):
                  t = torch.full((shape[0],), i * dt, device=device)
                  v = self.velocity(model, x, t, cond)
                  if method == "midpoint":
                        x_mid = x + 0.5 * dt * v
                        v = self.velocity(model, x_mid, t + 0.5 * dt, cond)
                  x = x + dt * v
                  traj.append(x.clone())
            return (x, traj) if return_traj else x
