import collections
import numpy as np
import torch

from toy_env import make_demos, sample_start
from flow_core import FlowMatcher
from ch03_train import (
      SequenceDataset, EMA, build_model, demos_to_episodes, run_episode,
      OBS_HORIZON, PRED_HORIZON, ACTION_HORIZON,
)


def train_flow(dataset, iters=3000, batch_size=128, lr=3e-4, time_sampling="beta", seed=0):
      torch.manual_seed(seed)
      loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
      model = build_model()
      ema = EMA(model)
      fm = FlowMatcher(time_sampling=time_sampling)
      opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
      sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=iters, pct_start=0.05)
      it = 0
      while it < iters:
            for obs, act in loader:
                  loss = fm.training_loss(model, act, obs.flatten(1))
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
      return ema.model, fm


class FlowPolicy:
      def __init__(self, model, fm, obs_norm, act_norm, steps=10):
            self.model = model
            self.fm = fm
            self.obs_norm = obs_norm
            self.act_norm = act_norm
            self.steps = steps
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
            x = self.fm.sample(self.model, (1, PRED_HORIZON, 2), cond, steps=self.steps)
            plan = self.act_norm.unnormalize(x[0].clamp(-1, 1).numpy())
            start = OBS_HORIZON - 1
            return plan[start:start + ACTION_HORIZON]


def evaluate_policy(policy, n=50, seed=1):
      rng = np.random.default_rng(seed)
      outcomes = collections.Counter()
      for _ in range(n):
            _, outcome = run_episode(policy, sample_start(rng))
            outcomes[outcome] += 1
      return dict(outcomes)


if __name__ == "__main__":
      dataset = SequenceDataset(demos_to_episodes(make_demos()))
      model, fm = train_flow(dataset)
      for steps in (1, 3, 10):
            torch.manual_seed(1)
            policy = FlowPolicy(model, fm, dataset.obs_norm, dataset.act_norm, steps=steps)
            print(f"flow policy {steps:2d} Euler steps: {evaluate_policy(policy)}")
      torch.save(model.state_dict(), "flow_policy.pt")
