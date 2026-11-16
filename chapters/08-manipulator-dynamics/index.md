---
title: "8. Manipulator Dynamics"
nav_order: 9
has_children: true
---

# 8. Manipulator Dynamics
{: .no_toc }

<p class="rlb-meta">Prerequisites: Chapter 7, Newton's laws, a first course in ordinary differential equations</p>

1. TOC
{:toc}

## What dynamics adds

Chapter 7 related joint angles to hand poses and joint velocities to hand twists, without asking what makes anything move. Dynamics answers that question. Given the joint torques $$\tau$$, how does the arm accelerate? Given a desired acceleration, what torques produce it? The first is **forward dynamics**, which a simulator integrates. The second is **inverse dynamics**, which a model based controller evaluates every millisecond. Both come from one equation, derived on this page. The split between a kinematic model and a dynamic model, and the state space view in which forward dynamics is just $$\dot x = f(x, u)$$ with $$x = (q, \dot q)$$, follow Zardini, Lorenzetti and Pavone [3].

Tedrake's observation is that this equation always has the same shape, whatever the arm [1]. Once you know the shape and its properties, you rarely need to look inside the individual terms.

## The Lagrangian route

### Generalised coordinates and energy

The joint angles $$q$$ are **generalised coordinates**. They describe the arm completely and independently, with no constraint left between them. Lagrangian mechanics works directly in such coordinates and never computes the internal forces in the joints, which is why it suits manipulators.

Consider link $$i$$ with mass $$m_i$$, centre of mass $$c_i$$, and rotational inertia $$\mathcal{I}_i$$ about its centre of mass, expressed in the link frame. Its kinetic energy is the sum of a translational and a rotational term. Both velocities are linear in $$\dot q$$ through Jacobians of the kind built in Section 7.1, $$\dot c_i = J_{v,i}(q)\,\dot q$$ and $$\omega_i = J_{\omega,i}(q)\,\dot q$$. Summing over links,

$$
T = \sum_i \tfrac12 m_i \lVert \dot c_i \rVert^2 + \tfrac12 \omega_i^\top R_i \mathcal{I}_i R_i^\top \omega_i
= \tfrac12\, \dot q^\top M(q)\, \dot q,
\qquad
M(q) = \sum_i m_i J_{v,i}^\top J_{v,i} + J_{\omega,i}^\top R_i \mathcal{I}_i R_i^\top J_{\omega,i} .
\tag{8.1}\label{eq:mass-matrix}
$$

The matrix $$M(q)$$ is the **mass matrix** or inertia matrix. The potential energy in uniform gravity $$\mathbf{g}$$ is $$U(q) = -\sum_i m_i\, \mathbf{g}^\top c_i(q)$$.

### Euler Lagrange equations

With the Lagrangian $$L = T - U$$, the equations of motion are [4, 8]

$$
\frac{d}{dt}\frac{\partial L}{\partial \dot q_k} - \frac{\partial L}{\partial q_k} = \tau_k, \qquad k = 1, \dots, n .
$$

Tedrake suggests reading them as a generalisation of Newton's law that you can crank like a handle [1]. Carry out the derivatives on $$T = \tfrac12 \dot q^\top M \dot q$$. The first term gives $$M \ddot q$$ plus terms in $$\dot M \dot q$$. The second gives terms in $$\partial M / \partial q$$, quadratic in $$\dot q$$, and the gradient of $$U$$. Collecting them,

$$
M(q)\,\ddot q + C(q, \dot q)\,\dot q + g(q) = \tau ,
\tag{8.2}\label{eq:manipulator}
$$

the **manipulator equation**. The entries of $$C$$ are usually written with Christoffel symbols of the mass matrix [8],

$$
C_{kj}(q, \dot q) = \sum_{i} \tfrac12\left(\frac{\partial M_{kj}}{\partial q_i} + \frac{\partial M_{ki}}{\partial q_j} - \frac{\partial M_{ij}}{\partial q_k}\right)\dot q_i ,
\qquad
g(q) = \frac{\partial U}{\partial q} .
\tag{8.3}\label{eq:christoffel}
$$

Products $$\dot q_i \dot q_j$$ with $$i \ne j$$ are **Coriolis** terms and squares $$\dot q_i^2$$ are **centrifugal** terms. We write $$h(q, \dot q) = C(q,\dot q)\,\dot q + g(q)$$ for the bias torques, everything except inertia.

{: .note }
> Sign conventions differ between sources. Tedrake moves gravity to the right hand side as $$\tau_g(q) = -g(q)$$ and writes $$M\ddot q + C\dot q = \tau_g + B u$$, where $$B$$ maps actuator inputs to joints [1]. The ETH notes write $$M\ddot q + b + g = \tau$$ [2]. In this book $$g(q)$$ is the torque the motors must supply to hold the arm still.

### Worked example: the double pendulum

Two links in a vertical plane, both joints about the same horizontal axis, point masses $$m_1, m_2$$ at the link ends, lengths $$l_1, l_2$$, and angles measured from the horizontal. The mass at the end of link 2 sits at $$l_1(\cos q_1, \sin q_1) + l_2(\cos(q_1 + q_2), \sin(q_1+q_2))$$ in the plane. Differentiating, squaring and collecting terms gives

$$
M(q) = \begin{bmatrix}
(m_1 + m_2) l_1^2 + m_2 l_2^2 + 2 m_2 l_1 l_2 \cos q_2 & m_2 l_2^2 + m_2 l_1 l_2 \cos q_2 \\
m_2 l_2^2 + m_2 l_1 l_2 \cos q_2 & m_2 l_2^2
\end{bmatrix} .
\tag{8.4}\label{eq:double-pendulum}
$$

The structure is already visible. $$M$$ depends on the elbow angle and not on the shoulder angle, because rotating the whole arm about the shoulder does not change how mass is distributed around it. The off diagonal term couples the joints. Accelerating the elbow pushes back on the shoulder.

The Lagrangian derivation works well by hand for two joints. For seven it produces expressions with thousands of terms, and a direct Lagrangian evaluation costs on the order of $$n^4$$ operations [5]. Section 8.1 computes the same quantities numerically in time linear in $$n$$.

## Properties worth knowing

These properties are what controllers and simulators actually use [1, 2, 6].

| Property | Statement | Why it matters |
|:--|:--|:--|
| Symmetric positive definite [7] | $$M = M^\top \succ 0$$ | Kinetic energy is positive for any motion. Forward dynamics $$\ddot q = M^{-1}(\tau - h)$$ always exists |
| Bounded | $$\mu_1 I \preceq M(q) \preceq \mu_2 I$$ for revolute arms | Gains can be tuned once for the whole workspace |
| Skew symmetry | $$\dot M - 2C$$ is skew symmetric when $$C$$ uses equation \eqref{eq:christoffel} | Gives the energy balance below and the stability proofs of Section 8.2 |
| $$C$$ is not unique | many matrices give the same vector $$C\dot q$$ | Only the Christoffel choice has the skew property |
| Linear in parameters | $$\tau = Y(q, \dot q, \ddot q)\,\pi$$ with 10 inertial parameters per link | System identification is linear least squares [9] |

The skew symmetry has a physical meaning. Differentiate the total energy along a motion and substitute equation \eqref{eq:manipulator}.

$$
\frac{d}{dt}\big(T + U\big) = \dot q^\top M \ddot q + \tfrac12 \dot q^\top \dot M \dot q + \dot q^\top g
= \dot q^\top \tau + \tfrac12\,\dot q^\top \big(\dot M - 2C\big)\,\dot q
= \dot q^\top \tau .
\tag{8.5}\label{eq:passivity}
$$

The last term vanishes because $$x^\top S x = 0$$ for any skew matrix $$S$$. The rate of change of energy equals the power delivered by the motors. The arm is a **passive** system. It cannot create energy, and a controller that only dissipates energy cannot destabilise it.

## Motors are part of the arm

Real joints are driven through gearboxes. A motor rotor with inertia $$I_m$$ behind a gear ratio $$N$$ adds a **reflected inertia** $$N^2 I_m$$ to the diagonal of $$M$$, often called the armature. With ratios around 100, this term can exceed the link inertia of the wrist joints by two orders of magnitude.

The numerical check in Section 8.1 shows how much this matters for our arm, with link masses close to the iiwa 14 and illustrative armature values.

```text
iiwa M(q_nom), links only    : symmetric gap 2.2e-16  eigenvalues 0.0005 to 2.915  condition number 5412
      diag M(q_nom): [1.573 2.626 0.421 0.522 0.01  0.007 0.001]
iiwa M(q_nom), links + rotors: symmetric gap 2.2e-16  eigenvalues 0.0505 to 3.300  condition number 65
      diag M(q_nom): [1.973 3.026 0.671 0.772 0.11  0.107 0.051]
```

Without rotors the last wrist joint has an inertia of $$10^{-3}$$ kg m², and the mass matrix has a condition number above 5000. An explicit integrator with a 1 ms step and any reasonable damping on that joint becomes unstable. With rotors the condition number drops to 65. Simulators expose this as the `armature` parameter in MuJoCo [10] and as rotor inertia with a gear ratio in Drake. Leaving it at zero is a common reason for a simulated arm to jitter.

Friction completes the joint model, usually as a viscous term $$b\,\dot q$$ and a Coulomb term $$f_c\,\mathrm{sign}(\dot q)$$ added to the left side of equation \eqref{eq:manipulator}.

{: .key }
> Every rigid serial arm obeys $$M(q)\ddot q + C(q,\dot q)\dot q + g(q) = \tau$$. $$M$$ is symmetric positive definite, $$\dot M - 2C$$ is skew symmetric, and together they say the arm only changes its energy through the work its motors do.

## Contact forces

When the hand pushes on the world with a wrench $$\mathcal{F}$$, equation (7.9) says the joints feel $$J^\top \mathcal{F}$$. The equation of motion gains one term [1].

$$
M(q)\,\ddot q + C(q, \dot q)\,\dot q + g(q) = \tau + J(q)^\top \mathcal{F}_{\text{ext}}
\tag{8.6}\label{eq:contact}
$$

This is the starting point for force control and for every contact model in a physics engine.

## References

1. R. Tedrake. *Underactuated Robotics.* Course notes for MIT 6.832, 2024. Appendix "Multi-Body Dynamics". [underactuated.mit.edu/multibody.html](https://underactuated.mit.edu/multibody.html)
2. Robotic Systems Lab, ETH Zurich. *Robot Dynamics Lecture Notes.* HS 2017, Chapter 3. [PDF](https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2017/RD_HS2017script.pdf)
3. G. Zardini, J. Lorenzetti, M. Pavone. *Principles of Robot Autonomy.* Stanford lecture notes, 2024. Section 1.2, kinematics and dynamics. [PDF](https://stanfordasl.github.io/PoRA-I/aa174a_aut2425/pdfs/AA_PoRA_lecture_notes_2425.pdf)
4. K. M. Lynch, F. C. Park. *Modern Robotics.* Cambridge University Press, 2017. Chapter 8. [modernrobotics.org](http://modernrobotics.org)
5. R. Featherstone. *Rigid Body Dynamics Algorithms.* Springer, 2008.
6. J.-J. E. Slotine, W. Li. *Applied Nonlinear Control.* Prentice Hall, 1991. Chapter 9.
7. H. Asada, J.-J. E. Slotine. *Robot Analysis and Control.* Wiley, 1986.
8. M. W. Spong, S. Hutchinson, M. Vidyasagar. *Robot Modeling and Control.* 2nd edition, Wiley, 2020. Chapter 6.
9. C. G. Atkeson, C. H. An, J. M. Hollerbach. *Estimation of Inertial Parameters of Manipulator Loads and Links.* International Journal of Robotics Research 5(3), 1986.
10. E. Todorov, T. Erez, Y. Tassa. *MuJoCo: A Physics Engine for Model-Based Control.* IROS 2012.
