import numpy as np

START = np.array([0.0, -1.0])
BRANCH = np.array([0.0, -0.75])
GOAL = np.array([0.0, 1.0])
OBSTACLE_CENTER = np.array([0.0, 0.0])
OBSTACLE_RADIUS = 0.4
STRAIGHT_LEN = 8
CURVE_LEN = 42


def bezier(start, control, goal, n):
      t = np.linspace(0.0, 1.0, n)[:, None]
      return (1 - t) ** 2 * start + 2 * (1 - t) * t * control + t ** 2 * goal


def sample_start(rng):
      return START + np.array([0.0, rng.normal(0.0, 0.03)])


def make_demos(n_demos=200, seed=0):
      rng = np.random.default_rng(seed)
      demos = []
      for _ in range(n_demos):
            start = sample_start(rng)
            side = rng.choice([-1.0, 1.0])
            via = np.array([side * rng.uniform(0.6, 0.8), rng.normal(0.0, 0.05)])
            control = 2.0 * via - 0.5 * (BRANCH + GOAL)
            straight = np.linspace(start, BRANCH, STRAIGHT_LEN, endpoint=False)
            curve = bezier(BRANCH, control, GOAL, CURVE_LEN)
            demos.append(np.concatenate([straight, curve]))
      return demos


def demos_to_pairs(demos):
      obs = np.concatenate([d[:-1] for d in demos]).astype(np.float32)
      act = np.concatenate([d[1:] - d[:-1] for d in demos]).astype(np.float32)
      return obs, act


def collided(pos):
      return np.linalg.norm(pos - OBSTACLE_CENTER) < OBSTACLE_RADIUS


def reached(pos, tol=0.1):
      return np.linalg.norm(pos - GOAL) < tol


def rollout(policy_step, start, max_steps=80):
      pos = np.array(start, dtype=np.float64)
      path = [pos.copy()]
      for _ in range(max_steps):
            for a in np.atleast_2d(policy_step(pos)):
                  pos = pos + a
                  path.append(pos.copy())
                  if collided(pos):
                        return np.array(path), "collision"
                  if reached(pos):
                        return np.array(path), "success"
                  if len(path) > max_steps:
                        return np.array(path), "timeout"
      return np.array(path), "timeout"


def evaluate(policy_step, n=50, seed=1):
      rng = np.random.default_rng(seed)
      results = {"success": 0, "collision": 0, "timeout": 0}
      paths = []
      for _ in range(n):
            start = sample_start(rng)
            path, outcome = rollout(policy_step, start)
            results[outcome] += 1
            paths.append(path)
      return results, paths
