import math


def wilson_interval(successes, trials, z=1.96):
      if trials == 0:
            return 0.0, 1.0
      p = successes / trials
      denom = 1 + z ** 2 / trials
      centre = (p + z ** 2 / (2 * trials)) / denom
      half = z * math.sqrt(p * (1 - p) / trials + z ** 2 / (4 * trials ** 2)) / denom
      return max(0.0, centre - half), min(1.0, centre + half)


if __name__ == "__main__":
      for s, n in [(45, 50), (39, 50), (9, 10), (18, 20)]:
            lo, hi = wilson_interval(s, n)
            print(f"{s:2d}/{n:2d}  rate {s / n:.2f}  95% interval [{lo:.2f}, {hi:.2f}]")
