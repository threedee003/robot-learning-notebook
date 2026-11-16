import numpy as np

DH_D = np.array([0.36, 0.0, 0.42, 0.0, 0.40, 0.0, 0.126])
DH_A = np.zeros(7)
DH_ALPHA = np.array([-1, 1, 1, -1, -1, 1, 0]) * np.pi / 2
Q_MAX = np.deg2rad([170, 120, 170, 120, 170, 120, 175])
Q_MIN = -Q_MAX
QD_MAX = np.deg2rad([85, 85, 100, 75, 130, 135, 135])
Q_NOMINAL = np.deg2rad([0, 30, 0, -60, 0, 90, 0])


def hat(w):
      return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def exp_so3(w, theta):
      K = hat(w)
      return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def log_so3(R):
      c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
      theta = np.arccos(c)
      if theta < 1e-9:
            return np.zeros(3)
      if np.pi - theta < 1e-6:
            k = np.argmax(np.diag(R))
            w = R[:, k] + np.eye(3)[k]
            return theta * w / np.linalg.norm(w)
      w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
      return theta * w / (2 * np.sin(theta))


def exp_se3(S, theta):
      w, v = S[:3], S[3:]
      T = np.eye(4)
      K = hat(w)
      T[:3, :3] = exp_so3(w, theta)
      G = np.eye(3) * theta + (1 - np.cos(theta)) * K + (theta - np.sin(theta)) * K @ K
      T[:3, 3] = G @ v
      return T


def inv_T(T):
      R, p = T[:3, :3], T[:3, 3]
      Ti = np.eye(4)
      Ti[:3, :3] = R.T
      Ti[:3, 3] = -R.T @ p
      return Ti


def adjoint(T):
      R, p = T[:3, :3], T[:3, 3]
      A = np.zeros((6, 6))
      A[:3, :3] = R
      A[3:, 3:] = R
      A[3:, :3] = hat(p) @ R
      return A


def dh(theta, d, a, alpha):
      ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
      return np.array([
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0, sa, ca, d],
            [0, 0, 0, 1],
      ])


def dh_frames(q):
      frames = [np.eye(4)]
      for i in range(7):
            frames.append(frames[-1] @ dh(q[i], DH_D[i], DH_A[i], DH_ALPHA[i]))
      return frames


def fk_dh(q):
      return dh_frames(q)[-1]


def screw_axes():
      frames = dh_frames(np.zeros(7))
      S = np.zeros((6, 7))
      for i in range(7):
            w = frames[i][:3, 2]
            p = frames[i][:3, 3]
            S[:3, i] = w
            S[3:, i] = -np.cross(w, p)
      return S, frames[-1]


S_LIST, M_HOME = screw_axes()


def fk(q, S=S_LIST, M=M_HOME):
      T = np.eye(4)
      for i in range(len(q)):
            T = T @ exp_se3(S[:, i], q[i])
      return T @ M


def jacobian_space(q, S=S_LIST):
      n = len(q)
      J = np.zeros((6, n))
      T = np.eye(4)
      for i in range(n):
            J[:, i] = adjoint(T) @ S[:, i]
            T = T @ exp_se3(S[:, i], q[i])
      return J


def jacobian_world(q):
      Js = jacobian_space(q)
      p = fk(q)[:3, 3]
      J = Js.copy()
      J[3:] = Js[3:] - hat(p) @ Js[:3]
      return J


def pose_error(T, T_des):
      e = np.zeros(6)
      e[:3] = log_so3(T_des[:3, :3] @ T[:3, :3].T)
      e[3:] = T_des[:3, 3] - T[:3, 3]
      return e


def random_config(rng, margin=0.9):
      return rng.uniform(margin * Q_MIN, margin * Q_MAX)
