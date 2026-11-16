import numpy as np

from kin_core import (S_LIST, M_HOME, Q_NOMINAL, fk, fk_dh, jacobian_world,
                      pose_error, random_config)


def numeric_jacobian(q, eps=1e-6):
      J = np.zeros((6, len(q)))
      T0 = fk(q)
      for i in range(len(q)):
            dq = np.zeros(len(q))
            dq[i] = eps
            J[:, i] = pose_error(T0, fk(q + dq)) / eps
      return J


def manipulability(J):
      return float(np.prod(np.linalg.svd(J, compute_uv=False)))


if __name__ == "__main__":
      rng = np.random.default_rng(0)
      np.set_printoptions(precision=3, suppress=True)

      print("screw axes S_i = (w_i, v_i) at q = 0:")
      print(S_LIST.T)
      print("home pose M:")
      print(M_HOME)

      dh_gap, jac_gap = 0.0, 0.0
      for _ in range(1000):
            q = random_config(rng)
            dh_gap = max(dh_gap, np.abs(fk(q) - fk_dh(q)).max())
            jac_gap = max(jac_gap, np.abs(jacobian_world(q) - numeric_jacobian(q)).max())
      print(f"max |FK_PoE - FK_DH| over 1000 configs: {dh_gap:.2e}")
      print(f"max |J - J_finite_diff| over 1000 configs: {jac_gap:.2e}")

      tests = {
            "nominal": Q_NOMINAL,
            "elbow straight q4=0": Q_NOMINAL * np.array([1, 1, 1, 0, 1, 1, 1]),
            "wrist aligned q6=0": Q_NOMINAL * np.array([1, 1, 1, 1, 1, 0, 1]),
            "shoulder aligned q2=0": Q_NOMINAL * np.array([1, 0, 1, 1, 1, 1, 1]),
            "all zero": np.zeros(7),
      }
      print(f"{'configuration':24s} {'rank':>4s} {'sigma_min':>10s} {'sigma_max':>10s} {'w(q)':>10s}")
      for name, q in tests.items():
            J = jacobian_world(q)
            s = np.linalg.svd(J, compute_uv=False)
            rank = int((s > 1e-9).sum())
            print(f"{name:24s} {rank:4d} {s[-1]:10.2e} {s[0]:10.3f} {manipulability(J):10.2e}")
