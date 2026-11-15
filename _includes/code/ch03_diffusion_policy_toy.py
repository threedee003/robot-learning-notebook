import numpy as np
import torch

from toy_env import make_demos, evaluate, BRANCH
from diffusion_core import NoiseSchedule
from ch02_ddpm_2d import Denoiser

PRED_HORIZON = 8
ACTION_HORIZON = 4


def chunk_dataset(demos, horizon=PRED_HORIZON):
      obs, chunks = [], []
      for d in demos:
            deltas = np.diff(d, axis=0)
            padded = np.concatenate([deltas, np.zeros((horizon, 2))], axis=0)
            for i in range(len(deltas)):
                  obs.append(d[i])
                  chunks.append(padded[i:i + horizon])
      return np.array(obs, dtype=np.float32), np.array(chunks, dtype=np.float32)


def train(obs, chunks, iters=6000, seed=0):
      torch.manual_seed(seed)
      scale = float(np.abs(chunks).max())
      obs_t = torch.from_numpy(obs)
      act_t = torch.from_numpy(chunks).flatten(1) / scale
      schedule = NoiseSchedule(num_steps=100)
      model = Denoiser(x_dim=act_t.shape[1], cond_dim=2)
      opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
      for it in range(iters):
            idx = torch.randint(0, len(obs_t), (256,))
            loss = schedule.training_loss(model, act_t[idx], obs_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
      return model.eval(), schedule, scale


def make_policy(model, schedule, scale, sampler="ddpm", steps=10):
      @torch.no_grad()
      def step(pos):
            cond = torch.tensor(pos, dtype=torch.float32)[None]
            shape = (1, PRED_HORIZON * 2)
            if sampler == "ddpm":
                  x = schedule.ddpm_sample(model, shape, cond).clamp(-1, 1)
            else:
                  x = schedule.ddim_sample(model, shape, cond, num_inference_steps=steps, clip=1.0)
            chunk = x.view(PRED_HORIZON, 2).numpy() * scale
            return chunk[:ACTION_HORIZON]
      return step


@torch.no_grad()
def fork_modes(model, schedule, scale, sampler, n=500):
      cond = torch.tensor(BRANCH, dtype=torch.float32)[None].repeat(n, 1)
      if sampler == "ddpm":
            x = schedule.ddpm_sample(model, (n, PRED_HORIZON * 2), cond)
      else:
            x = schedule.ddim_sample(model, (n, PRED_HORIZON * 2), cond, num_inference_steps=10, clip=1.0)
      lateral = x.view(n, PRED_HORIZON, 2)[:, :ACTION_HORIZON, 0].sum(1) * scale
      return int((lateral < -0.1).sum()), int((lateral.abs() <= 0.1).sum()), int((lateral > 0.1).sum())


if __name__ == "__main__":
      demos = make_demos()
      obs, chunks = chunk_dataset(demos)
      model, schedule, scale = train(obs, chunks)
      for sampler in ("ddpm", "ddim"):
            torch.manual_seed(1)
            results, paths = evaluate(make_policy(model, schedule, scale, sampler))
            left, middle, right = fork_modes(model, schedule, scale, sampler)
            print(f"{sampler}: rollouts {results}  fork samples left {left} middle {middle} right {right}")
      torch.save({"model": model.state_dict(), "scale": scale}, "dp_toy.pt")
