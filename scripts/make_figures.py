import os
import sys
import shutil
import subprocess
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "_includes", "code")
MEDIA = os.path.join(ROOT, "assets", "media")
sys.path.insert(0, CODE)

import toy_env as te
from ch01_bc_mse import train_bc
from ch02_ddpm_2d import Denoiser, eight_gaussians
from diffusion_core import NoiseSchedule
from ch03_diffusion_policy_toy import chunk_dataset, train as train_dp, make_policy
from ch04_cfm_2d import train_cfm
from ch04_ot_cfm import minibatch_ot

LEFT = "#2563eb"
RIGHT = "#ea580c"
FAIL = "#dc2626"
GREY = "#9ca3af"
INK = "#111827"

plt.rcParams.update({
      "figure.dpi": 150,
      "savefig.dpi": 150,
      "font.size": 10,
      "axes.edgecolor": INK,
      "axes.labelcolor": INK,
      "xtick.color": INK,
      "ytick.color": INK,
      "axes.spines.top": False,
      "axes.spines.right": False,
})


def out(*parts):
      path = os.path.join(MEDIA, *parts)
      os.makedirs(os.path.dirname(path), exist_ok=True)
      return path


def arena(ax, title=None):
      ax.add_patch(Circle(te.OBSTACLE_CENTER, te.OBSTACLE_RADIUS, color=GREY, alpha=0.6, lw=0))
      ax.plot(*te.START, "o", color=INK, ms=5)
      ax.plot(*te.GOAL, "*", color=INK, ms=11)
      ax.set_xlim(-1.2, 1.2)
      ax.set_ylim(-1.15, 1.15)
      ax.set_aspect("equal")
      ax.set_xticks([])
      ax.set_yticks([])
      for spine in ax.spines.values():
            spine.set_visible(False)
      if title:
            ax.set_title(title, fontsize=10)


def side_color(path):
      return LEFT if path[np.argmax(np.abs(path[:, 0])), 0] < 0 else RIGHT


def plot_rollouts(ax, paths):
      for p in paths:
            failed = te.collided(p[-1]) or not te.reached(p[-1])
            ax.plot(p[:, 0], p[:, 1], color=FAIL if failed else side_color(p), lw=0.9, alpha=0.7)
            if failed:
                  ax.plot(*p[-1], "x", color=FAIL, ms=4)


def fig_demos():
      demos = te.make_demos()
      fig, ax = plt.subplots(figsize=(4, 4))
      arena(ax)
      for d in demos:
            ax.plot(d[:, 0], d[:, 1], color=side_color(d), lw=0.6, alpha=0.35)
      ax.plot(*te.BRANCH, "o", color=INK, ms=6, mfc="white", mew=1.5)
      ax.annotate("fork", te.BRANCH, xytext=(0.25, -0.95), arrowprops=dict(arrowstyle="->", color=INK), color=INK)
      fig.tight_layout()
      fig.savefig(out("ch01", "demos.png"), facecolor="white")
      plt.close(fig)


def fig_bc():
      obs, act = te.demos_to_pairs(te.make_demos())
      fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.9))
      for ax, epochs in zip(axes, (100, 300)):
            policy, scale = train_bc(obs, act, epochs=epochs)

            @torch.no_grad()
            def step(pos):
                  return policy(torch.tensor(pos, dtype=torch.float32)[None])[0].numpy() * scale

            results, paths = te.evaluate(step)
            arena(ax, f"MSE, {epochs} epochs: {results['success']}/50 succeed")
            plot_rollouts(ax, paths)
      fig.tight_layout()
      fig.savefig(out("ch01", "bc_rollouts.png"), facecolor="white")
      plt.close(fig)


def fig_ddpm_animation():
      torch.manual_seed(0)
      schedule = NoiseSchedule(100)
      model = Denoiser()
      opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
      for _ in range(4000):
            x0, _ = eight_gaussians(512)
            loss = schedule.training_loss(model, x0)
            opt.zero_grad()
            loss.backward()
            opt.step()
      model.eval()
      _, traj = schedule.ddpm_sample(model, (1500, 2), return_traj=True)
      frames_dir = out("ch02", "_frames", "x")
      frames_dir = os.path.dirname(frames_dir)
      os.makedirs(frames_dir, exist_ok=True)
      keep = list(range(0, 101, 2)) + [100] * 12
      for i, j in enumerate(keep):
            k = 100 - j
            fig, ax = plt.subplots(figsize=(4, 4))
            x = traj[j].numpy()
            ax.scatter(x[:, 0], x[:, 1], s=2, color=LEFT, alpha=0.6, lw=0)
            ax.set_xlim(-3.2, 3.2)
            ax.set_ylim(-3.2, 3.2)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"reverse diffusion, k = {k}", fontsize=10)
            for spine in ax.spines.values():
                  spine.set_visible(False)
            fig.tight_layout()
            fig.savefig(os.path.join(frames_dir, f"f{i:03d}.png"), facecolor="white")
            plt.close(fig)
            if j == 100 and i == len(keep) - 1:
                  shutil.copy(os.path.join(frames_dir, f"f{i:03d}.png"), out("ch02", "ddpm_final.png"))
      pattern = os.path.join(frames_dir, "f%03d.png")
      subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "12", "-i", pattern,
                      "-vf", "scale=600:-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                      "-pix_fmt", "yuv420p", "-movflags", "+faststart", out("ch02", "ddpm_reverse.mp4")], check=True)
      subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "12", "-i", pattern,
                      "-vf", "scale=360:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse",
                      out("ch02", "ddpm_reverse.gif")], check=True)
      shutil.rmtree(frames_dir)


def fig_dp_toy():
      obs, chunks = chunk_dataset(te.make_demos())
      model, schedule, scale = train_dp(obs, chunks)
      fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.9))
      for ax, (sampler, label) in zip(axes, (("ddpm", "DDPM, 100 steps"), ("ddim", "DDIM, 10 steps"))):
            torch.manual_seed(1)
            results, paths = te.evaluate(make_policy(model, schedule, scale, sampler))
            arena(ax, f"Diffusion Policy, {label}: {results['success']}/50")
            plot_rollouts(ax, paths)
      fig.tight_layout()
      fig.savefig(out("ch03", "dp_toy_rollouts.png"), facecolor="white")
      plt.close(fig)


def fig_flow_paths():
      fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.9))
      for ax, (coupling, label) in zip(axes, ((None, "independent coupling"), (minibatch_ot, "minibatch OT coupling"))):
            model, fm = train_cfm(coupling=coupling)
            torch.manual_seed(7)
            _, traj = fm.sample(model, (300, 2), steps=50, return_traj=True)
            t = torch.stack(traj).numpy()
            for i in range(t.shape[1]):
                  ax.plot(t[:, i, 0], t[:, i, 1], color=LEFT, lw=0.5, alpha=0.35)
            ax.scatter(t[0, :, 0], t[0, :, 1], s=3, color=GREY, lw=0)
            ax.scatter(t[-1, :, 0], t[-1, :, 1], s=4, color=RIGHT, lw=0)
            ax.set_xlim(-3.2, 3.2)
            ax.set_ylim(-3.2, 3.2)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(label, fontsize=10)
            for spine in ax.spines.values():
                  spine.set_visible(False)
      fig.tight_layout()
      fig.savefig(out("ch04", "flow_paths.png"), facecolor="white")
      plt.close(fig)


if __name__ == "__main__":
      jobs = {
            "demos": fig_demos,
            "bc": fig_bc,
            "ddpm": fig_ddpm_animation,
            "dp": fig_dp_toy,
            "flow": fig_flow_paths,
      }
      selected = sys.argv[1:] or list(jobs)
      for name in selected:
            print("making", name, flush=True)
            jobs[name]()
