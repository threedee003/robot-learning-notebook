---
title: "8.1 Recursive Newton Euler"
parent: "8. Manipulator Dynamics"
nav_order: 1
---

# 8.1 Recursive Newton Euler
{: .no_toc }

1. TOC
{:toc}

## One rigid body

Start with a single body, in a frame attached at its centre of mass. Let its twist be $$\mathcal{V} = (\omega, v)$$ in that body frame. Newton's law for the centre of mass, written in a rotating frame, picks up a term from the rotation of the frame. Euler's law for the rotation picks up the gyroscopic term $$\omega \times \mathcal{I}\omega$$.

$$
f = m\,(\dot v + \omega \times v), \qquad
m_{\text{torque}} = \mathcal{I}\,\dot\omega + \omega \times \mathcal{I}\,\omega
$$

Stack force and moment into a **wrench** $$\mathcal{F} = (m_{\text{torque}}, f)$$ and mass and inertia into the $$6 \times 6$$ **spatial inertia** $$\mathcal{G} = \mathrm{diag}(\mathcal{I}, m I_3)$$. Both laws then fit in one line [2, 4].

$$
\mathcal{F} = \mathcal{G}\,\dot{\mathcal{V}} - \mathrm{ad}_{\mathcal{V}}^\top\,\mathcal{G}\,\mathcal{V},
\qquad
\mathrm{ad}_{\mathcal{V}} = \begin{bmatrix} [\omega] & 0 \\ [v] & [\omega] \end{bmatrix}
\tag{8.7}\label{eq:newton-euler}
$$

Expanding the product confirms it. $$\mathrm{ad}_{\mathcal{V}}^\top \mathcal{G}\mathcal{V} = (-\omega \times \mathcal{I}\omega - v \times m v,\; -m\,\omega \times v)$$ and $$v \times v = 0$$, so the two rows are Euler's and Newton's equations.

## A chain of bodies

A serial arm is a chain of such bodies, each pushed by its parent through a joint and pulling on its child through the next. The **recursive Newton Euler algorithm** of Luh, Walker and Paul [6] computes inverse dynamics, $$\tau = \mathrm{ID}(q, \dot q, \ddot q)$$, in two sweeps.

**Forward sweep, base to hand.** The twist of link $$i$$ is its parent's twist moved into link $$i$$'s frame, plus the contribution of joint $$i$$. Here $$\mathcal{A}_i$$ is the screw axis of joint $$i$$ expressed in link $$i$$'s frame and $$X_{i,i-1}$$ is the transform from the parent frame to the link frame, which depends on $$q_i$$.

$$
\mathcal{V}_i = \mathrm{Ad}_{X_{i,i-1}}\mathcal{V}_{i-1} + \mathcal{A}_i\,\dot q_i,
\qquad
\dot{\mathcal{V}}_i = \mathrm{Ad}_{X_{i,i-1}}\dot{\mathcal{V}}_{i-1} + \mathrm{ad}_{\mathcal{V}_i}\mathcal{A}_i\,\dot q_i + \mathcal{A}_i\,\ddot q_i
\tag{8.8}\label{eq:forward-sweep}
$$

The middle term of the acceleration is the velocity product, the same $$\omega \times$$ term as before, now arising because the joint axis itself moves with the parent.

**Backward sweep, hand to base.** The wrench that joint $$i$$ must transmit is what link $$i$$ needs for its own motion, equation \eqref{eq:newton-euler}, plus whatever it passes on to link $$i+1$$. The joint torque is the part of that wrench along the joint axis. The rest is carried by the bearing.

$$
\mathcal{F}_i = \mathrm{Ad}_{X_{i+1,i}}^\top \mathcal{F}_{i+1} + \mathcal{G}_i\,\dot{\mathcal{V}}_i - \mathrm{ad}_{\mathcal{V}_i}^\top \mathcal{G}_i\,\mathcal{V}_i,
\qquad
\tau_i = \mathcal{A}_i^\top \mathcal{F}_i
\tag{8.9}\label{eq:backward-sweep}
$$

Gravity costs nothing extra. Set the base acceleration to $$\dot{\mathcal{V}}_0 = (0, -\mathbf{g})$$. Accelerating the whole arm upward at $$9.81$$ m/s² in free space produces the same joint torques as holding it still in gravity, by the equivalence principle. A hand wrench enters as the initial value $$\mathcal{F}_{n+1}$$ of the backward sweep.

{: .definition }
> **Inverse dynamics** maps $$(q, \dot q, \ddot q)$$ to $$\tau$$. **Forward dynamics** maps $$(q, \dot q, \tau)$$ to $$\ddot q$$. RNEA computes the first in $$O(n)$$ operations.

## Everything else from RNEA

The manipulator equation is linear in $$\ddot q$$, so one inverse dynamics routine gives every term of it.

| Quantity | Call | Cost |
|:--|:--|:--|
| Bias $$h = C\dot q + g$$ | $$\mathrm{ID}(q, \dot q, 0)$$ | 1 call |
| Gravity $$g(q)$$ | $$\mathrm{ID}(q, 0, 0)$$ | 1 call |
| Coriolis $$C\dot q$$ | $$\mathrm{ID}(q, \dot q, 0)$$ with $$\mathbf{g} = 0$$ | 1 call |
| Column $$j$$ of $$M$$ | $$\mathrm{ID}(q, 0, e_j)$$ with $$\mathbf{g} = 0$$ | $$n$$ calls for all of $$M$$ |
| Forward dynamics | solve $$M\ddot q = \tau - h$$ by Cholesky | $$n + 1$$ calls and an $$O(n^3)$$ solve |

For $$n = 7$$ this is fast enough. Production libraries replace the column trick with the **composite rigid body algorithm**, which builds $$M$$ in $$O(n^2)$$ with a much smaller constant [7], and compute forward dynamics directly with Featherstone's **articulated body algorithm** in $$O(n)$$ [5]. Pinocchio [9], Drake and MuJoCo all implement these. Tedrake's notes point to the same literature and stress exploiting the sparsity of $$M$$ for branched robots [1].

{: .note }
> Featherstone's spatial vector algebra and the Lie group notation used here, from Lynch and Park [4], describe the same objects. Featherstone orders spatial vectors with the angular part first too, but defines force and motion vectors as dual spaces with their own cross products. When you port code from one book to the other, check the ordering and the sign of each cross product term.

## Implementation

`dyn_core.py` implements equations \eqref{eq:forward-sweep} and \eqref{eq:backward-sweep} for any serial arm described by its screw axes, centre of mass frames and spatial inertias. Link inertias are modelled as solid cylinders. The rotor armature from the chapter introduction is added to the diagonal. `make_double_pendulum` builds the two link arm of equation (8.4) in the same framework, so it can be checked against the closed form.

{% include codefile.html file="dyn_core.py" %}

`ch08_dynamics.py` runs four checks.

{% include codefile.html file="ch08_dynamics.py" %}

```text
double pendulum, RNEA vs closed form: max |dM| 8.9e-16  max |dg| 7.1e-15
g(q_nom) [N m]: [ -0.    -35.326  -0.     15.559   0.     -0.      0.   ]
qd^T Mdot qd = 2 qd^T C(q,qd) qd, worst relative gap over 50 states: 9.3e-09
euler          dt 1 ms, 2 s of free motion: energy drift +1.3e+00 J  max |drift| 1.3e+00 J
semi-implicit  dt 1 ms, 2 s of free motion: energy drift +2.2e-01 J  max |drift| 3.7e-01 J
rk4            dt 1 ms, 2 s of free motion: energy drift -1.7e-09 J  max |drift| 1.7e-09 J
```

**Closed form.** RNEA reproduces the double pendulum mass matrix and gravity torques to machine precision.

**Gravity at the nominal posture.** Only the shoulder (joint 2) and the elbow (joint 4) carry load, 35 and 16 N m. The vertical joints 1, 3, 5 and 7 cannot be turned by gravity in this posture. Joint 6 sees no torque because the hand hangs straight down below it.

**Energy identity.** Equation (8.5) requires $$\dot q^\top \dot M \dot q = 2\,\dot q^\top C \dot q$$. The test estimates $$\dot M$$ by a central difference along $$\dot q$$ and compares it with the Coriolis vector from RNEA. The gap is at the level of the finite difference error. This identity holds for the vector $$C\dot q$$, which is unique, so it does not depend on which matrix $$C$$ one picks.

**Energy conservation.** With $$\tau = 0$$ and no friction, the total energy of a falling arm is constant. Any change is integration error.

{% include figure.html src="/assets/media/ch08/energy_drift.png" alt="Energy drift of three integrators for a freely falling arm" caption="Absolute energy error of a freely falling 7 joint arm, 1 ms step. Explicit Euler gains energy steadily. Semi implicit Euler drifts less but is not bounded here. RK4 stays below $$10^{-8}$$ J." width="75%" %}

Explicit Euler always adds energy to an oscillating system, which is why it is never used alone for mechanics. Semi implicit Euler, the default in MuJoCo, updates velocity first and then position with the new velocity. It conserves a nearby energy exactly for separable Hamiltonians, $$H = T(p) + U(q)$$ [8]. A manipulator's kinetic energy depends on $$q$$ through $$M(q)$$, so the guarantee is lost and we see a drift of 0.2 J in 2 s. Physics engines hide this with joint damping and small steps. RK4 is accurate but costs four dynamics evaluations per step. Zardini, Lorenzetti and Pavone discuss the same trade off between integrator order and cost for simulating robot models [3].

{: .warning }
> An energy check is the cheapest test of a dynamics implementation. A wrong sign in the velocity product of equation \eqref{eq:forward-sweep}, or a missing gyroscopic term, still produces motion that looks plausible in a video. It does not conserve energy.

{: .exercise }
> Remove `ad(V[i + 1]) @ self.A[i] * qd[i]` from the forward sweep and rerun `ch08_dynamics.py`. Which checks fail, and which still pass? Explain why the mass matrix check is blind to this bug.

## References

1. R. Tedrake. *Underactuated Robotics.* Appendix "Multi-Body Dynamics", section on recursive algorithms. [underactuated.mit.edu/multibody.html](https://underactuated.mit.edu/multibody.html)
2. Robotic Systems Lab, ETH Zurich. *Robot Dynamics Lecture Notes.* HS 2017, Section 3.3 on the Newton Euler method. [PDF](https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2017/RD_HS2017script.pdf)
3. G. Zardini, J. Lorenzetti, M. Pavone. *Principles of Robot Autonomy.* Stanford lecture notes, 2024. Section 1.4 on simulating robot dynamics. [PDF](https://stanfordasl.github.io/PoRA-I/aa174a_aut2425/pdfs/AA_PoRA_lecture_notes_2425.pdf)
4. K. M. Lynch, F. C. Park. *Modern Robotics.* Cambridge University Press, 2017. Chapter 8.3. [modernrobotics.org](http://modernrobotics.org)
5. R. Featherstone. *Rigid Body Dynamics Algorithms.* Springer, 2008.
6. J. Y. S. Luh, M. W. Walker, R. P. C. Paul. *On-Line Computational Scheme for Mechanical Manipulators.* Journal of Dynamic Systems, Measurement, and Control 102(2), 1980.
7. M. W. Walker, D. E. Orin. *Efficient Dynamic Computer Simulation of Robotic Mechanisms.* Journal of Dynamic Systems, Measurement, and Control 104(3), 1982.
8. E. Hairer, C. Lubich, G. Wanner. *Geometric Numerical Integration.* 2nd edition, Springer, 2006.
9. J. Carpentier, G. Saurel, G. Buondonno, J. Mirabel, F. Lamiraux, O. Stasse, N. Mansard. *The Pinocchio C++ Library.* IEEE/SICE International Symposium on System Integration, 2019.
