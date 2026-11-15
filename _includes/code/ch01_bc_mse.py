import numpy as np
import torch
import torch.nn as nn

from toy_env import make_demos, demos_to_pairs, evaluate, BRANCH


class MLPPolicy(nn.Module):
      def __init__(self, obs_dim=2, act_dim=2, hidden=256):
            super().__init__()
            self.net = nn.Sequential(
                  nn.Linear(obs_dim, hidden), nn.ReLU(),
                  nn.Linear(hidden, hidden), nn.ReLU(),
                  nn.Linear(hidden, act_dim),
            )

      def forward(self, obs):
            return self.net(obs)


def train_bc(obs, act, epochs=300, lr=1e-3, seed=0):
      torch.manual_seed(seed)
      obs_t = torch.from_numpy(obs)
      act_scale = float(np.abs(act).max())
      act_t = torch.from_numpy(act) / act_scale
      policy = MLPPolicy()
      opt = torch.optim.Adam(policy.parameters(), lr=lr)
      loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(obs_t, act_t), batch_size=256, shuffle=True
      )
      for epoch in range(epochs):
            for o, a in loader:
                  loss = ((policy(o) - a) ** 2).mean()
                  opt.zero_grad()
                  loss.backward()
                  opt.step()
      return policy, act_scale


if __name__ == "__main__":
      demos = make_demos()
      obs, act = demos_to_pairs(demos)
      at_fork = np.all(np.isclose(obs, BRANCH), axis=1)
      print("demonstrated actions at the fork (first 4):")
      print(act[at_fork][:4].round(3))

      for epochs in (100, 300):
            policy, act_scale = train_bc(obs, act, epochs=epochs)

            @torch.no_grad()
            def step(pos):
                  o = torch.tensor(pos, dtype=torch.float32)[None]
                  return policy(o)[0].numpy() * act_scale

            results, _ = evaluate(step)
            print(f"epochs {epochs}: action at fork {step(BRANCH).round(3)}  rollouts {results}")
