import copy
import collections
import numpy as np
import torch

from toy_env import make_demos, collided, reached, sample_start
from diffusion_core import NoiseSchedule
from ch03_unet import ConditionalUnet1D

OBS_HORIZON = 2
PRED_HORIZON = 16
ACTION_HORIZON = 8


def demos_to_episodes(demos):
      return [{"obs": d[:-1].astype(np.float32), "action": d[1:].astype(np.float32)} for d in demos]


class MinMaxNormalizer:
      def __init__(self, data):
            self.lo = data.min(axis=0)
            self.hi = data.max(axis=0)

      def normalize(self, x):
            return 2.0 * (x - self.lo) / (self.hi - self.lo + 1e-8) - 1.0

      def unnormalize(self, x):
            return (x + 1.0) / 2.0 * (self.hi - self.lo + 1e-8) + self.lo


class SequenceDataset(torch.utils.data.Dataset):
      def __init__(self, episodes):
            all_obs = np.concatenate([e["obs"] for e in episodes])
            all_act = np.concatenate([e["action"] for e in episodes])
            self.obs_norm = MinMaxNormalizer(all_obs)
            self.act_norm = MinMaxNormalizer(all_act)
            self.episodes = [
                  {"obs": self.obs_norm.normalize(e["obs"]), "action": self.act_norm.normalize(e["action"])}
                  for e in episodes
            ]
            self.index = [(i, t) for i, e in enumerate(self.episodes) for t in range(len(e["obs"]))]

      def __len__(self):
            return len(self.index)

      def __getitem__(self, k):
            i, t = self.index[k]
            e = self.episodes[i]
            n = len(e["obs"])
            first = t - OBS_HORIZON + 1
            obs_idx = np.clip(np.arange(first, t + 1), 0, n - 1)
            act_idx = np.clip(np.arange(first, first + PRED_HORIZON), 0, n - 1)
            return torch.from_numpy(e["obs"][obs_idx]), torch.from_numpy(e["action"][act_idx])


class EMA:
      def __init__(self, model, decay=0.995):
            self.model = copy.deepcopy(model).eval()
            self.decay = decay
            for p in self.model.parameters():
                  p.requires_grad_(False)

      @torch.no_grad()
      def update(self, model):
            for pe, pm in zip(self.model.parameters(), model.parameters()):
                  pe.mul_(self.decay).add_(pm.detach(), alpha=1 - self.decay)
            for be, bm in zip(self.model.buffers(), model.buffers()):
                  be.copy_(bm)


def build_model(obs_dim=2, act_dim=2):
      return ConditionalUnet1D(action_dim=act_dim, global_cond_dim=OBS_HORIZON * obs_dim, step_embed_dim=32, dims=(32, 64, 128))


def train(dataset, iters=3000, batch_size=128, lr=3e-4, seed=0):
      torch.manual_seed(seed)
      loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
      model = build_model()
      ema = EMA(model)
      schedule = NoiseSchedule(num_steps=100)
      opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
      sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=iters, pct_start=0.05)
      it = 0
      while it < iters:
            for obs, act in loader:
                  loss = schedule.training_loss(model, act, obs.flatten(1))
                  opt.zero_grad()
                  loss.backward()
                  opt.step()
                  sched.step()
                  ema.update(model)
                  if it % 500 == 0:
                        print(f"iter {it:5d}  loss {loss.item():.4f}")
                  it += 1
                  if it >= iters:
                        break
      return ema.model, schedule


class DiffusionPolicy:
      def __init__(self, model, schedule, obs_norm, act_norm, inference_steps=10):
            self.model = model
            self.schedule = schedule
            self.obs_norm = obs_norm
            self.act_norm = act_norm
            self.inference_steps = inference_steps
            self.history = collections.deque(maxlen=OBS_HORIZON)

      def reset(self, obs):
            self.history.clear()
            for _ in range(OBS_HORIZON):
                  self.history.append(obs)

      @torch.no_grad()
      def act(self, obs):
            self.history.append(obs)
            o = self.obs_norm.normalize(np.stack(self.history)).astype(np.float32)
            cond = torch.from_numpy(o).flatten()[None]
            x = self.schedule.ddim_sample(self.model, (1, PRED_HORIZON, 2), cond, self.inference_steps, clip=1.0)
            plan = self.act_norm.unnormalize(x[0].numpy())
            start = OBS_HORIZON - 1
            return plan[start:start + ACTION_HORIZON]


def run_episode(policy, start, max_steps=80):
      pos = np.array(start, dtype=np.float64)
      policy.reset(pos)
      path = [pos.copy()]
      while len(path) <= max_steps:
            for target in policy.act(pos):
                  pos = target.astype(np.float64)
                  path.append(pos.copy())
                  if collided(pos):
                        return np.array(path), "collision"
                  if reached(pos):
                        return np.array(path), "success"
      return np.array(path), "timeout"


if __name__ == "__main__":
      dataset = SequenceDataset(demos_to_episodes(make_demos()))
      model, schedule = train(dataset)
      policy = DiffusionPolicy(model, schedule, dataset.obs_norm, dataset.act_norm)
      torch.manual_seed(1)
      rng = np.random.default_rng(1)
      outcomes = collections.Counter()
      for _ in range(50):
            _, outcome = run_episode(policy, sample_start(rng))
            outcomes[outcome] += 1
      print("UNet Diffusion Policy:", dict(outcomes))
