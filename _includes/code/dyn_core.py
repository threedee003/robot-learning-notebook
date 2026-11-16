import numpy as np

from kin_core import S_LIST, M_HOME, hat, exp_se3, inv_T, adjoint

GRAVITY = np.array([0.0, 0.0, -9.81])
LINK_MASS = np.array([4.0, 4.0, 3.0, 2.7, 1.7, 1.8, 0.3])
LINK_COM_Z = np.array([0.28, 0.46, 0.66, 0.88, 1.06, 1.20, 1.28])
LINK_LENGTH = np.array([0.20, 0.20, 0.20, 0.20, 0.20, 0.10, 0.05])
LINK_RADIUS = 0.06
ARMATURE = np.array([0.40, 0.40, 0.25, 0.25, 0.10, 0.10, 0.05])


def translation(p):
      T = np.eye(4)
      T[:3, 3] = p
      return T


def spatial_inertia(m, I):
      G = np.zeros((6, 6))
      G[:3, :3] = I
      G[3:, 3:] = m * np.eye(3)
      return G


def cylinder_inertia(m, r, L):
      ixx = m * (3 * r ** 2 + L ** 2) / 12
      return np.diag([ixx, ixx, 0.5 * m * r ** 2])


def ad(V):
      A = np.zeros((6, 6))
      A[:3, :3] = hat(V[:3])
      A[3:, :3] = hat(V[3:])
      A[3:, 3:] = hat(V[:3])
      return A


class SerialArm:
      def __init__(self, S, M_com, M_ee, G, armature=None):
            self.S = S
            self.M_com = M_com
            self.n = S.shape[1]
            self.G = G
            self.M_rel = [M_com[0]]
            for i in range(1, self.n):
                  self.M_rel.append(inv_T(M_com[i - 1]) @ M_com[i])
            self.M_rel.append(inv_T(M_com[-1]) @ M_ee)
            self.A = [adjoint(inv_T(M_com[i])) @ S[:, i] for i in range(self.n)]
            self.mass = np.array([g[3, 3] for g in G])
            self.armature = np.zeros(self.n) if armature is None else armature

      def rnea(self, q, qd, qdd, g=GRAVITY, F_tip=np.zeros(6)):
            n = self.n
            V = [np.zeros(6)]
            Vd = [np.concatenate([np.zeros(3), -g])]
            Ad = []
            for i in range(n):
                  T = exp_se3(-self.A[i], q[i]) @ inv_T(self.M_rel[i])
                  Ad.append(adjoint(T))
                  V.append(Ad[i] @ V[i] + self.A[i] * qd[i])
                  Vd.append(Ad[i] @ Vd[i] + self.A[i] * qdd[i] + ad(V[i + 1]) @ self.A[i] * qd[i])
            Ad.append(adjoint(inv_T(self.M_rel[n])))
            F = F_tip.copy()
            tau = np.zeros(n)
            for i in reversed(range(n)):
                  F = Ad[i + 1].T @ F + self.G[i] @ Vd[i + 1] - ad(V[i + 1]).T @ (self.G[i] @ V[i + 1])
                  tau[i] = F @ self.A[i]
            return tau + self.armature * qdd

      def mass_matrix(self, q):
            z = np.zeros(self.n)
            return np.column_stack([self.rnea(q, z, e, g=np.zeros(3)) for e in np.eye(self.n)])

      def coriolis(self, q, qd):
            return self.rnea(q, qd, np.zeros(self.n), g=np.zeros(3))

      def gravity(self, q):
            z = np.zeros(self.n)
            return self.rnea(q, z, z)

      def forward_dynamics(self, q, qd, tau):
            h = self.rnea(q, qd, np.zeros(self.n))
            return np.linalg.solve(self.mass_matrix(q), tau - h)

      def com_positions(self, q):
            T = np.eye(4)
            out = []
            for i in range(self.n):
                  T = T @ exp_se3(self.S[:, i], q[i])
                  out.append((T @ self.M_com[i])[:3, 3])
            return np.array(out)

      def energy(self, q, qd, g=GRAVITY):
            kinetic = 0.5 * qd @ self.mass_matrix(q) @ qd
            potential = -np.sum(self.mass * (self.com_positions(q) @ g))
            return kinetic + potential


def make_iiwa(mass_scale=1.0, armature=ARMATURE):
      M_com = [translation([0, 0, z]) for z in LINK_COM_Z]
      G = [spatial_inertia(mass_scale * m, cylinder_inertia(mass_scale * m, LINK_RADIUS, L))
           for m, L in zip(LINK_MASS, LINK_LENGTH)]
      return SerialArm(S_LIST, M_com, M_HOME, G, armature)


def make_double_pendulum(m1=1.0, m2=1.0, l1=1.0, l2=1.0):
      S = np.zeros((6, 2))
      S[:, 0] = [0, 1, 0, 0, 0, 0]
      S[:, 1] = [0, 1, 0, 0, 0, l1]
      M_com = [translation([l1, 0, 0]), translation([l1 + l2, 0, 0])]
      G = [spatial_inertia(m1, np.zeros((3, 3))), spatial_inertia(m2, np.zeros((3, 3)))]
      return SerialArm(S, M_com, translation([l1 + l2, 0, 0]), G)
