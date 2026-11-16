import numpy as np
from scipy.optimize import minimize

from kin_core import (Q_MIN, Q_MAX, QD_MAX, Q_NOMINAL, fk, jacobian_world,
                      pose_error, random_config)


def ik_step(J, e, method, lam, q, k_null):
      if method == "pinv":
            return np.linalg.pinv(J) @ e
      if method == "lm":
            lam = np.sqrt(e @ e + 1e-6)
      dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
      if method == "dls+null":
            P = np.eye(J.shape[1]) - np.linalg.pinv(J) @ J
            dq = dq + P @ (k_null * (Q_NOMINAL - q))
      return dq


def solve_ik(T_des, q0, method="dls", lam=0.05, k_null=0.1, iters=200,
             tol_pos=1e-4, tol_rot=1e-3, history=False):
      q = q0.copy()
      errs = []
      for it in range(iters):
            e = pose_error(fk(q), T_des)
            errs.append(np.linalg.norm(e))
            if np.linalg.norm(e[3:]) < tol_pos and np.linalg.norm(e[:3]) < tol_rot:
                  return (q, it, True, errs) if history else (q, it, True)
            q = np.clip(q + ik_step(jacobian_world(q), e, method, lam, q, k_null), Q_MIN, Q_MAX)
      return (q, iters, False, errs) if history else (q, iters, False)


def diff_ik_pinv(q, V_des):
      return np.linalg.pinv(jacobian_world(q)) @ V_des


def diff_ik_qp(q, V_des, h, eps=1e-3):
      n = len(q)
      J = jacobian_world(q)
      lo = np.maximum(-QD_MAX, (Q_MIN - q) / h)
      hi = np.minimum(QD_MAX, (Q_MAX - q) / h)
      cost = lambda x: -x[-1] + 0.5 * eps * x[:n] @ x[:n]
      grad = lambda x: np.concatenate([eps * x[:n], [-1.0]])
      cons = {"type": "eq", "fun": lambda x: J @ x[:n] - x[-1] * V_des,
              "jac": lambda x: np.hstack([J, -V_des[:, None]])}
      res = minimize(cost, np.zeros(n + 1), jac=grad, constraints=[cons],
                     bounds=list(zip(lo, hi)) + [(0.0, 1.0)], method="SLSQP")
      return res.x[:n], res.x[-1]


def teleop(controller, q0, V_des, h=0.01, T=6.0):
      q = q0.copy()
      log = []
      for k in range(int(T / h)):
            if controller == "pinv":
                  v, alpha = diff_ik_pinv(q, V_des), 1.0
            else:
                  v, alpha = diff_ik_qp(q, V_des, h)
            s_min = np.linalg.svd(jacobian_world(q), compute_uv=False)[-1]
            p = fk(q)[:3, 3]
            q = q + h * v
            jump = np.linalg.norm(fk(q)[:3, 3] - p)
            log.append((k * h, p[2], np.max(np.abs(v) / QD_MAX), alpha, s_min, jump))
      return np.array(log)


if __name__ == "__main__":
      rng = np.random.default_rng(0)
      targets = [fk(random_config(rng)) for _ in range(300)]
      print(f"{'method':10s} {'success':>8s} {'median iters':>13s} {'mean |q - q_nom|':>17s}")
      for method in ("pinv", "dls", "lm", "dls+null"):
            res = [solve_ik(T, Q_NOMINAL, method=method) for T in targets]
            ok = [r for r in res if r[2]]
            dist = np.mean([np.linalg.norm(r[0] - Q_NOMINAL) for r in ok])
            print(f"{method:10s} {len(ok):5d}/{len(res)} {np.median([r[1] for r in ok]):13.0f} {dist:17.3f}")

      T_hard = fk(np.deg2rad([40, 50, -30, 1, 20, -40, 10]))
      for method in ("pinv", "dls", "lm"):
            q, it, ok, errs = solve_ik(T_hard, Q_NOMINAL, method=method, history=True)
            print(f"near-singular target, {method:8s}: converged {ok}  iterations {it}")

      solved = 0
      for T in targets:
            for attempt in range(5):
                  q0 = Q_NOMINAL if attempt == 0 else random_config(rng)
                  if solve_ik(T, q0, method="lm")[2]:
                        solved += 1
                        break
      print(f"lm with up to 4 random restarts: {solved}/{len(targets)}")

      V_des = np.array([0, 0, 0, 0, 0, 0.10])
      q0 = np.deg2rad([0, 20, 0, -50, 0, 60, 0])
      for name in ("pinv", "qp"):
            log = teleop(name, q0, V_des)
            over = log[log[:, 2] > 1.0]
            t_over = f"{over[0, 0]:.2f} s" if len(over) else "never"
            print(f"{name:4s} final z {log[-1, 1]:.3f} m  max |v|/v_max {log[:, 2].max():7.2f}  "
                  f"limit first exceeded {t_over:>7s}  largest hand jump in one step "
                  f"{100 * log[:, 5].max():6.2f} cm  final alpha {log[-1, 3]:.2f}")
