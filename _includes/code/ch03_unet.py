import torch
import torch.nn as nn

from diffusion_core import SinusoidalEmbedding


class Conv1dBlock(nn.Module):
      def __init__(self, c_in, c_out, kernel=5, groups=8):
            super().__init__()
            self.block = nn.Sequential(
                  nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2),
                  nn.GroupNorm(groups, c_out),
                  nn.Mish(),
            )

      def forward(self, x):
            return self.block(x)


class FiLMResBlock(nn.Module):
      def __init__(self, c_in, c_out, cond_dim, kernel=5, groups=8):
            super().__init__()
            self.block1 = Conv1dBlock(c_in, c_out, kernel, groups)
            self.block2 = Conv1dBlock(c_out, c_out, kernel, groups)
            self.film = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, 2 * c_out))
            self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

      def forward(self, x, cond):
            h = self.block1(x)
            scale, bias = self.film(cond).unsqueeze(-1).chunk(2, dim=1)
            h = scale * h + bias
            h = self.block2(h)
            return h + self.skip(x)


class ConditionalUnet1D(nn.Module):
      def __init__(self, action_dim, global_cond_dim, step_embed_dim=128, dims=(256, 512, 1024), kernel=5, groups=8):
            super().__init__()
            self.step_embed = nn.Sequential(
                  SinusoidalEmbedding(step_embed_dim),
                  nn.Linear(step_embed_dim, 4 * step_embed_dim),
                  nn.Mish(),
                  nn.Linear(4 * step_embed_dim, step_embed_dim),
            )
            cond_dim = step_embed_dim + global_cond_dim
            chans = [action_dim, *dims]
            pairs = list(zip(chans[:-1], chans[1:]))

            self.down = nn.ModuleList()
            for i, (c_in, c_out) in enumerate(pairs):
                  is_last = i == len(pairs) - 1
                  self.down.append(nn.ModuleList([
                        FiLMResBlock(c_in, c_out, cond_dim, kernel, groups),
                        FiLMResBlock(c_out, c_out, cond_dim, kernel, groups),
                        nn.Conv1d(c_out, c_out, 3, stride=2, padding=1) if not is_last else nn.Identity(),
                  ]))

            self.mid = nn.ModuleList([
                  FiLMResBlock(dims[-1], dims[-1], cond_dim, kernel, groups),
                  FiLMResBlock(dims[-1], dims[-1], cond_dim, kernel, groups),
            ])

            self.up = nn.ModuleList()
            for c_in, c_out in reversed(pairs[1:]):
                  self.up.append(nn.ModuleList([
                        FiLMResBlock(2 * c_out, c_in, cond_dim, kernel, groups),
                        FiLMResBlock(c_in, c_in, cond_dim, kernel, groups),
                        nn.ConvTranspose1d(c_in, c_in, 4, stride=2, padding=1),
                  ]))

            self.final = nn.Sequential(
                  Conv1dBlock(dims[0], dims[0], kernel, groups),
                  nn.Conv1d(dims[0], action_dim, 1),
            )

      def forward(self, x, t, global_cond):
            x = x.transpose(1, 2)
            cond = torch.cat([self.step_embed(t), global_cond], dim=-1)
            skips = []
            for res1, res2, down in self.down:
                  x = res2(res1(x, cond), cond)
                  skips.append(x)
                  x = down(x)
            for res in self.mid:
                  x = res(x, cond)
            for res1, res2, up in self.up:
                  x = torch.cat([x, skips.pop()], dim=1)
                  x = res2(res1(x, cond), cond)
                  x = up(x)
            return self.final(x).transpose(1, 2)


if __name__ == "__main__":
      net = ConditionalUnet1D(action_dim=7, global_cond_dim=2 * 64)
      x = torch.randn(4, 16, 7)
      t = torch.randint(0, 100, (4,))
      cond = torch.randn(4, 2 * 64)
      print("output", tuple(net(x, t, cond).shape))
      print("params %.1fM" % (sum(p.numel() for p in net.parameters()) / 1e6))
