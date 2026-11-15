import numpy as np


class TemporalEnsembler:
      def __init__(self, chunk_len, action_dim, m=0.1, max_t=1000):
            self.chunk_len = chunk_len
            self.m = m
            self.buffer = np.full((max_t, max_t + chunk_len, action_dim), np.nan)
            self.t = 0

      def reset(self):
            self.buffer[:] = np.nan
            self.t = 0

      def step(self, chunk):
            self.buffer[self.t, self.t:self.t + self.chunk_len] = chunk
            preds = self.buffer[: self.t + 1, self.t]
            preds = preds[~np.isnan(preds).any(axis=1)]
            w = np.exp(-self.m * np.arange(len(preds)))
            w = w / w.sum()
            self.t += 1
            return (w[:, None] * preds).sum(axis=0)


if __name__ == "__main__":
      ens = TemporalEnsembler(chunk_len=4, action_dim=1, m=0.5, max_t=10)
      rng = np.random.default_rng(0)
      for k in range(6):
            chunk = np.arange(k, k + 4, dtype=float)[:, None] + rng.normal(0, 0.3, (4, 1))
            print(f"t={k}  executed {ens.step(chunk)[0]:.3f}  target {k}")
