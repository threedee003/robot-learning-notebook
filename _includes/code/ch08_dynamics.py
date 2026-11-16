import numpy as np

from dyn_core import make_iiwa, make_double_pendulum
from kin_core import Q_NOMINAL, random_config


def double_pendulum_check(m1=1.0, m2=2.0, l1=1.0, l2=0.5, g=9.81):
      arm = make_double_pendulum(m1, m2, l1, l2)
      rng = np.random.default_rng(1)
      err_M, err_g = 0.0, 0.0
      for _ in range(100):
            q = rng.uniform(-np.pi, np.pi, 2)
            c1, c2, c12 = np.cos(q[0]), np.cos(q[1]), np.cos(q[0] + q[1])
            M = np.array([
                  [(m1 + m2) * l1 ** 2 + m2 * l2 ** 2 + 2 * m2 * l1 * l2 * c2, m2 * l2 ** 2 + m2 * l1 * l2 * c2],
                  [m2 * l2 ** 2 + m2 * l1 * l2 * c2, m2 * l2 ** 2],
            ])
            tau_hold = -np.array([(m1 + m2) * g * l1 * c1 + m2 * g * l2 * c12, m2 * g * l2 * c12])
            err_M = max(err_M, np.abs(arm.mass_matrix(q) - M).max())
            err_g = max(err_g, np.abs(arm.gravity(q) - tau_hold).max())
      return err_M, err_g


def step(arm, q, qd, dt, method):
      f = lambda q_, qd_: arm.forward_dynamics(q_, qd_, np.zeros(arm.n))
      if method == "euler":
            return q + dt * qd, qd + dt * f(q, qd)
      if method == "semi-implicit":
            qd_new = qd + dt * f(q, qd)
            return q + dt * qd_new, qd_new
      k1q, k1v = qd, f(q, qd)
      k2q, k2v = qd + 0.5 * dt * k1v, f(q + 0.5 * dt * k1q, qd + 0.5 * dt * k1v)
      k3q, k3v = qd + 0.5 * dt * k2v, f(q + 0.5 * dt * k2q, qd + 0.5 * dt * k2v)
      k4q, k4v = qd + dt * k3v, f(q + dt * k3q, qd + dt * k3v)
      return (q + dt / 6 * (k1q + 2 * k2q + 2 * k3q + k4q),
              qd + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v))


def free_fall(arm, q0, method, dt=1e-3, T=2.0):
      q, qd = q0.copy(), np.zeros(arm.n)
      E0 = arm.energy(q, qd)
      drift = []
      for _ in range(int(T / dt)):
            q, qd = step(arm, q, qd, dt, method)
            drift.append(arm.energy(q, qd) - E0)
      return np.array(drift)


if __name__ == "__main__":
      np.set_printoptions(precision=3, suppress=True)
      err_M, err_g = double_pendulum_check()
      print(f"double pendulum, RNEA vs closed form: max |dM| {err_M:.1e}  max |dg| {err_g:.1e}")

      for label, arm in (("links only", make_iiwa(armature=np.zeros(7))), ("links + rotors", make_iiwa())):
            M = arm.mass_matrix(Q_NOMINAL)
            eig = np.linalg.eigvalsh(M)
            print(f"iiwa M(q_nom), {label:14s}: symmetric gap {np.abs(M - M.T).max():.1e}  "
                  f"eigenvalues {eig.min():.4f} to {eig.max():.3f}  condition number {eig.max() / eig.min():.0f}")
            print("      diag M(q_nom):", np.diag(M))
      print("g(q_nom) [N m]:", arm.gravity(Q_NOMINAL))

      rng = np.random.default_rng(0)
      worst = 0.0
      for _ in range(50):
            q, qd = random_config(rng), rng.normal(size=7)
            eps = 1e-6
            Md = (arm.mass_matrix(q + eps * qd) - arm.mass_matrix(q - eps * qd)) / (2 * eps)
            lhs = qd @ Md @ qd
            rhs = 2 * qd @ arm.coriolis(q, qd)
            worst = max(worst, abs(lhs - rhs) / abs(lhs))
      print(f"qd^T Mdot qd = 2 qd^T C(q,qd) qd, worst relative gap over 50 states: {worst:.1e}")

      q0 = np.deg2rad([0, 60, 0, -30, 0, 45, 0])
      for method in ("euler", "semi-implicit", "rk4"):
            d = free_fall(arm, q0, method)
            print(f"{method:14s} dt 1 ms, 2 s of free motion: energy drift {d[-1]:+.1e} J  max |drift| {np.abs(d).max():.1e} J")
