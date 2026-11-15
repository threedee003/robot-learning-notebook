import time
import torch

from diffusion_core import NoiseSchedule
from flow_core import FlowMatcher
from ch03_unet import ConditionalUnet1D


def time_call(fn, repeats=20):
      fn()
      t0 = time.perf_counter()
      for _ in range(repeats):
            fn()
      return 1000 * (time.perf_counter() - t0) / repeats


if __name__ == "__main__":
      torch.manual_seed(0)
      torch.set_num_threads(1)
      configs = {
            "toy U-Net (32, 64, 128), 2-D actions": dict(action_dim=2, global_cond_dim=4, step_embed_dim=32, dims=(32, 64, 128)),
            "paper U-Net (256, 512, 1024), 7-D actions": dict(action_dim=7, global_cond_dim=128, step_embed_dim=128, dims=(256, 512, 1024)),
      }
      schedule = NoiseSchedule(100)
      fm = FlowMatcher()
      for name, cfg in configs.items():
            model = ConditionalUnet1D(**cfg).eval()
            shape = (1, 16, cfg["action_dim"])
            cond = torch.randn(1, cfg["global_cond_dim"])
            with torch.no_grad():
                  rows = {
                        "DDPM 100 steps": lambda: schedule.ddpm_sample(model, shape, cond),
                        "DDIM 10 steps": lambda: schedule.ddim_sample(model, shape, cond, 10, clip=1.0),
                        "flow 10 Euler steps": lambda: fm.sample(model, shape, cond, steps=10),
                        "flow 3 Euler steps": lambda: fm.sample(model, shape, cond, steps=3),
                        "flow 1 Euler step": lambda: fm.sample(model, shape, cond, steps=1),
                  }
                  print(name)
                  for label, fn in rows.items():
                        print(f"   {label:20s} {time_call(fn, 5 if 'DDPM' in label else 20):8.1f} ms per chunk")
