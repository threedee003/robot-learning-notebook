import numpy as np

from dyn_core import make_iiwa
from kin_core import Q_NOMINAL

AMP = np.deg2rad([20, 15, 20, 20, 30, 30, 30])
FREQ = 0.5
OMEGA_N = 10.0


def reference(t):
      w = 2 * np.pi * FREQ
      phase = np.arange(7) * 0.5
      q = Q_NOMINAL + AMP * np.sin(w * t + phase)
      qd = AMP * w * np.cos(w * t + phase)
      qdd = -AMP * w ** 2 * np.sin(w * t + phase)
      return q, qd, qdd


class PD:
      def __init__(self, model, gravity_comp):
            m = np.diag(model.mass_matrix(Q_NOMINAL))
            self.Kp = OMEGA_N ** 2 * m
            self.Kd = 2 * OMEGA_N * m
            self.model = model
            self.gravity_comp = gravity_comp

      def __call__(self, q, qd, q_ref, qd_ref, qdd_ref):
            tau = self.Kp * (q_ref - q) + self.Kd * (qd_ref - qd)
            if self.gravity_comp:
                  tau = tau + self.model.gravity(q)
            return tau


class ComputedTorque:
      def __init__(self, model):
            self.model = model

      def __call__(self, q, qd, q_ref, qd_ref, qdd_ref):
            a = qdd_ref + OMEGA_N ** 2 * (q_ref - q) + 2 * OMEGA_N * (qd_ref - qd)
            return self.model.mass_matrix(q) @ a + self.model.rnea(q, qd, np.zeros(7))


def simulate(plant, controller, T=4.0, dt=1e-3, tau_max=None):
      q, qd, _ = reference(0.0)
      errors = []
      for k in range(int(T / dt)):
            t = k * dt
            q_ref, qd_ref, qdd_ref = reference(t)
            errors.append(q_ref - q)
            tau = controller(q, qd, q_ref, qd_ref, qdd_ref)
            if tau_max is not None:
                  tau = np.clip(tau, -tau_max, tau_max)
            qd = qd + dt * plant.forward_dynamics(q, qd, tau)
            q = q + dt * qd
      return np.array(errors)


if __name__ == "__main__":
      plant = make_iiwa()
      wrong = make_iiwa(mass_scale=0.8)
      runs = {
            "PD": PD(plant, gravity_comp=False),
            "PD + gravity": PD(plant, gravity_comp=True),
            "computed torque": ComputedTorque(plant),
            "computed torque, -20% mass": ComputedTorque(wrong),
      }
      print(f"{'controller':28s} {'RMS error [deg]':>16s} {'max error [deg]':>16s} {'worst joint':>12s}")
      for name, ctrl in runs.items():
            e = np.rad2deg(simulate(plant, ctrl))
            settled = e[1000:]
            rms = np.sqrt((settled ** 2).mean())
            worst = np.abs(settled).max(axis=0)
            print(f"{name:28s} {rms:16.3f} {worst.max():16.3f} {int(worst.argmax()) + 1:12d}")
