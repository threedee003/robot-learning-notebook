---
title: "7.1 Jacobians and Singularities"
parent: "7. Manipulator Kinematics"
nav_order: 1
---

# 7.1 Jacobians and Singularities
{: .no_toc }

1. TOC
{:toc}

## From joint velocities to hand velocity

Differentiate forward kinematics in time. Every joint contributes a twist proportional to its own speed, so the hand twist is linear in $$\dot q$$.

$$
\mathcal{V} = J(q)\,\dot q, \qquad J(q) \in \R^{6 \times n}
\tag{7.7}\label{eq:jac}
$$

Column $$i$$ of $$J$$ is the twist the hand would have if only joint $$i$$ moved at unit speed. The product of exponentials makes that column easy to write. Joint $$i$$ rotates about its screw axis $$\mathcal{S}_i$$, but that axis has itself been moved by joints $$1$$ to $$i-1$$. Moving a screw axis is an adjoint, equation (7.4), so

$$
J_{s,i}(q) = \mathrm{Ad}_{\,e^{[\mathcal{S}_1] q_1} \cdots e^{[\mathcal{S}_{i-1}] q_{i-1}}}\; \mathcal{S}_i ,
\qquad J_{s,1} = \mathcal{S}_1 .
\tag{7.8}\label{eq:space-jac}
$$

This is the **space Jacobian** [5]. Its twists are expressed in the base frame and their linear part is the velocity of the body point currently at the base origin. That point is rarely the one we care about. The velocity of the hand point $$p$$ follows from the same cross product rule as before, $$v_p = v_s + \omega \times p = v_s - [p]\,\omega$$. The resulting matrix, with angular rows from $$J_s$$ and linear rows giving the velocity of the hand point in world coordinates, is what Tedrake writes $$J^G$$ [1] and what most controllers use. In the code it is `jacobian_world`. Haviland and Corke derive the same columns from the elementary transform sequence and continue to second order, the manipulator Hessian, which is what you need for the gradient of manipulability [3].

## Geometric and analytic Jacobians

The ETH notes draw a distinction that causes real bugs [2]. The **geometric** Jacobian of equation \eqref{eq:jac} maps $$\dot q$$ to the angular velocity $$\omega$$. The **analytic** Jacobian maps $$\dot q$$ to the time derivative of some orientation parameters $$\chi$$, such as Euler angles. The two are related by a configuration dependent matrix, $$\omega = E(\chi)\,\dot\chi$$, so

$$
J_A(q) = \begin{bmatrix} E(\chi)^{-1} & 0 \\ 0 & I \end{bmatrix} J(q).
$$

The matrix $$E(\chi)$$ loses rank at the representation singularities listed in the rotation table of the chapter introduction, for example at gimbal lock. The analytic Jacobian can then be singular even though the arm is not. Use the geometric Jacobian unless you have a reason not to, and convert orientation errors to an angular velocity like quantity with the matrix logarithm.

## Checking a Jacobian

A Jacobian derived by hand is wrong more often than not. A finite difference check takes a few lines. Perturb one joint by $$\varepsilon$$, compute the pose error between the two poses as a 6 vector (rotation error from the logarithm, then position error), and divide by $$\varepsilon$$.

{% include codefile.html file="ch07_kinematics.py" %}

```text
max |J - J_finite_diff| over 1000 configs: 4.68e-07
configuration            rank  sigma_min  sigma_max       w(q)
nominal                     6   1.65e-01      1.833   1.14e-01
elbow straight q4=0         5   2.81e-17      1.896   1.65e-17
wrist aligned q6=0          6   1.11e-01      1.932   3.74e-02
shoulder aligned q2=0       6   1.65e-01      1.833   7.13e-02
all zero                    3   0.00e+00      2.000   0.00e+00
```

The gap of $$5 \times 10^{-7}$$ is the expected truncation error of a forward difference with $$\varepsilon = 10^{-6}$$, of order $$\varepsilon$$ times the second derivative. A sign error or a wrong frame would give a gap of order 1.

## Singularities

### What the singular values mean

Take the singular value decomposition $$J = U \Sigma V^\top$$. The unit ball of joint velocities, $$\lVert \dot q \rVert \le 1$$, maps to an ellipsoid of hand twists whose semi axes point along the columns of $$U$$ and have lengths $$\sigma_1 \ge \dots \ge \sigma_6$$. The hand moves easily along the long axes and badly along the short ones. When $$\sigma_6 = 0$$ the ellipsoid is flat and there is a direction in which the hand cannot move at all, at first order. That configuration is a **kinematic singularity**.

Two scalar summaries are common. The smallest singular value $$\sigma_{\min}$$ tells you how close you are to losing a direction. Yoshikawa's **manipulability** $$w(q) = \sqrt{\det(JJ^\top)} = \prod_i \sigma_i$$ measures the volume of the ellipsoid [6]. Volume can stay large while one axis collapses, so for safety checks $$\sigma_{\min}$$ is the better number.

{: .note }
> $$\sigma_{\min}$$ mixes units. The angular rows are in rad/s per rad/s and the linear rows in m/s per rad/s. Scaling the linear rows by a characteristic length, for example the arm reach, changes which singular value is smallest. Report the scaling you used.

### Which configurations are singular, and why

The table above contains a lesson about redundancy. Setting $$q_6 = 0$$ lines up the axes of joints 5 and 7, and setting $$q_2 = 0$$ lines up joints 1 and 3. Two parallel collinear axes produce the same column of $$J$$, so one column is wasted. A 6 joint arm would become singular. Our arm has a seventh column to spare and keeps rank 6.

The elbow is different. Let $$S$$ be the shoulder point where axes 1, 2 and 3 meet, and $$W$$ the wrist point where axes 5, 6 and 7 meet. The distance between them depends on $$q_4$$ alone,

$$
\lVert W - S \rVert^2 = l_u^2 + l_f^2 + 2\, l_u l_f \cos q_4, \qquad l_u = 0.42,\; l_f = 0.40 .
$$

Its derivative with respect to $$q_4$$ is proportional to $$\sin q_4$$, which vanishes at $$q_4 = 0$$. No joint can change the shoulder to wrist distance at first order when the elbow is straight, so the wrist cannot move along the line through the shoulder. The redundant joint does not help, because joints 1 to 3 only rotate that line about $$S$$ and joints 5 to 7 sit beyond $$W$$. The rank drops to 5.

{% include figure.html src="/assets/media/ch07/elbow_singularity.png" alt="Smallest singular value and manipulability against elbow angle" caption="Smallest singular value and manipulability as the elbow sweeps through straight. Both fall linearly to zero at $$q_4 = 0$$, as $$\sin q_4$$ predicts." width="85%" %}

At $$q = 0$$ the arm is a vertical stick. Joints 1, 3, 5 and 7 all spin about the same vertical line and joints 2, 4 and 6 all tip it about parallel horizontal axes. Only 3 independent twists remain.

{: .key }
> A singularity is a configuration where the Jacobian loses rank and the hand loses a direction of motion. Redundancy removes some singularities, those caused by aligned axes, but not those caused by the geometry of the chain, such as a straight elbow.

Tedrake stresses that the singularity at full extension is a physical limit [1]. At full extension the hand truly cannot move further outward at that instant. The robot can still pass through the configuration, for example by bending the elbow the other way, but a velocity controller that insists on the impossible direction will command infinite joint speeds. Section 7.2 shows this happen.

## Forces and the Jacobian transpose

Kinematics also tells us how forces map. Let the hand push on the environment with a wrench $$\mathcal{F} = (m, f)$$, a moment and a force. By the principle of virtual work, the power delivered by the joints equals the power delivered at the hand for every motion, $$\tau^\top \dot q = \mathcal{F}^\top \mathcal{V} = \mathcal{F}^\top J \dot q$$. Since this holds for every $$\dot q$$ [4, 5],

$$
\tau = J(q)^\top \mathcal{F}.
\tag{7.9}\label{eq:statics}
$$

At a singularity, a hand force along the lost direction needs no joint torque at all. The structure carries it. This is why you lock your elbow to hold a heavy box with straight arms. The force ellipsoid has the same axes as the velocity ellipsoid with inverted lengths, so good directions for speed are bad directions for force and the reverse. Equation \eqref{eq:statics} returns in Chapter 8, where it maps contact forces into the equations of motion.

{: .exercise }
> Scale the linear rows of $$J$$ by $$1/1.306$$, the reach of the arm, and recompute the table. Does the ordering of the configurations by $$\sigma_{\min}$$ change?

## References

1. R. Tedrake. *Robotic Manipulation.* Chapter "Basic Pick and Place", sections on differential kinematics and invertibility of the Jacobian. [manipulation.mit.edu/pick.html](https://manipulation.mit.edu/pick.html)
2. Robotic Systems Lab, ETH Zurich. *Robot Dynamics Lecture Notes.* HS 2017, Section 2.8 on analytical and geometric Jacobians. [PDF](https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2017/RD_HS2017script.pdf)
3. J. Haviland, P. Corke. *Manipulator Differential Kinematics, Part I.* IEEE Robotics and Automation Magazine, 2023. [arXiv:2207.01796](https://arxiv.org/abs/2207.01796)
4. B. Siciliano et al. *Robotics: Modelling, Planning and Control.* Springer, 2009. Chapter 3.
5. K. M. Lynch, F. C. Park. *Modern Robotics.* Cambridge University Press, 2017. Chapter 5. [modernrobotics.org](http://modernrobotics.org)
6. T. Yoshikawa. *Manipulability of Robotic Mechanisms.* International Journal of Robotics Research 4(2), 1985.
