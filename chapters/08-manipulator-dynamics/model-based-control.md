---
title: "8.2 Model-Based Control"
parent: "8. Manipulator Dynamics"
nav_order: 2
---

# 8.2 Model-Based Control
{: .no_toc }

1. TOC
{:toc}

## Where this sits under a learned policy

A policy from Chapter 3 or 5 emits joint or hand targets at 10 to 50 Hz. A joint torque controller underneath runs at around 1 kHz and turns those targets into motor commands. How well the arm follows its targets depends on that controller, and a policy trained in simulation with one controller and deployed with another meets a different robot. This section derives the standard controllers from the manipulator equation and compares them on the arm of Section 8.1. Tedrake builds the same controllers in Drake in his chapter on force control [1].

## PD control and gravity compensation

The simplest controller ignores the model and acts like a spring and damper on each joint, $$\tau = K_p e + K_d \dot e$$ with $$e = q_{\text{ref}} - q$$. At rest, gravity must be balanced by the spring, so the arm settles with a steady error $$e_{ss} = K_p^{-1} g(q)$$. Stiffer gains shrink it but never remove it.

Adding the gravity torque removes the steady error for set point regulation.

$$
\tau = K_p\,(q_{\text{ref}} - q) - K_d\,\dot q + g(q)
\tag{8.10}\label{eq:pd-gravity}
$$

Takegaki and Arimoto proved that this is globally asymptotically stable for any positive definite gains [5]. The proof is short and uses the passivity of Chapter 8. Take the candidate energy $$V = \tfrac12 \dot q^\top M \dot q + \tfrac12 e^\top K_p e$$, kinetic energy plus the energy stored in the virtual spring. Along the closed loop, with a constant $$q_{\text{ref}}$$,

$$
\dot V = \dot q^\top M \ddot q + \tfrac12 \dot q^\top \dot M \dot q - \dot q^\top K_p e
= \dot q^\top\big(K_p e - K_d \dot q - C\dot q\big) + \tfrac12 \dot q^\top \dot M \dot q - \dot q^\top K_p e
= -\dot q^\top K_d\, \dot q \le 0 .
$$

The skew symmetry of $$\dot M - 2C$$ cancelled the Coriolis terms. $$\dot V = 0$$ only when $$\dot q = 0$$, and the only rest state of the closed loop with $$\dot q = 0$$ is $$e = 0$$, so LaSalle's invariance principle gives convergence [6]. Notice what the proof did not need. It never used the values of $$M$$ or $$C$$, only their structure.

## Computed torque

To **track** a moving reference, the controller must also supply the torques that accelerate the arm along it. Computed torque, also called inverse dynamics control or feedback linearisation, chooses

$$
\tau = M(q)\,\big(\ddot q_{\text{ref}} + K_d \dot e + K_p e\big) + C(q, \dot q)\,\dot q + g(q) .
\tag{8.11}\label{eq:computed-torque}
$$

Substitute into the manipulator equation. $$M$$ is invertible, so the nonlinear terms cancel exactly and each joint obeys a decoupled linear equation,

$$
\ddot e + K_d\,\dot e + K_p\,e = 0 .
$$

With $$K_p = \omega_n^2 I$$ and $$K_d = 2\zeta\omega_n I$$, every joint behaves like a mass spring damper with natural frequency $$\omega_n$$ and damping ratio $$\zeta$$, anywhere in the workspace. The bracket in equation \eqref{eq:computed-torque} is the desired acceleration, and RNEA evaluates the whole right hand side in one call when given that acceleration. Tedrake presents the same cancellation as feedback linearisation and shows how much of it survives when some joints have no motor [3]. The structure, a feedforward term from the model plus feedback on the error, is the same one used for trajectory tracking in mobile robots [10].

The cancellation is only as good as the model. With estimates $$\hat M, \hat h$$ the closed loop becomes

$$
\ddot e + K_d\,\dot e + K_p\,e = \hat M^{-1}\big[(M - \hat M)\,\ddot q + (h - \hat h)\big],
$$

a linear system driven by a disturbance proportional to the model error. Adaptive control estimates the inertial parameters online, using the linear parameterisation of Chapter 8, and again relies on skew symmetry for its stability proof [7].

## Experiment

All seven joints of the arm from Section 8.1 track sinusoids of 15 to 30 degrees amplitude at 0.5 Hz around $$q_{\text{nom}}$$, with a different phase per joint. The controller and the simulation run at 1 kHz with semi implicit Euler. The PD gains are $$K_p = \omega_n^2 \,\mathrm{diag}(M(q_{\text{nom}}))$$ and $$K_d = 2\omega_n\,\mathrm{diag}(M(q_{\text{nom}}))$$ with $$\omega_n = 10$$ rad/s, scaled by each joint's inertia so that every joint gets a similar bandwidth. Computed torque uses $$\omega_n = 10$$ and $$\zeta = 1$$. The last run gives the controller a model whose link masses and inertias are 20% too small. The errors exclude the first second.

{% include codefile.html file="ch08_control.py" %}

```text
controller                    RMS error [deg]  max error [deg]  worst joint
PD                                      4.896           11.210            4
PD + gravity                            1.809            4.174            3
computed torque                         0.015            0.027            5
computed torque, -20% mass              0.756            1.982            4
```

{% include figure.html src="/assets/media/ch08/tracking.png" alt="Joint tracking error for four controllers" caption="Norm of the joint error over time. Each model term removes roughly an order of magnitude of error. A 20% mass error costs most of the benefit of computed torque but still beats PD with gravity compensation." %}

Three lessons.

1. Plain PD is worst at the elbow, joint 4, which carries the largest gravity load relative to its gain.
2. Gravity compensation helps by a factor of 2.7. What remains is the inertial torque the PD terms must generate from error, since nothing feeds forward the reference acceleration.
3. Exact computed torque reduces the error by another factor of more than 100. The residual 0.015 degrees comes from discretisation. The controller holds each torque for 1 ms while the state moves. With a 20% wrong model the error grows fifty times, which is why system identification matters on hardware [8].

{: .key }
> Gravity compensation gets you regulation. Feedforward of the full inverse dynamics gets you tracking. Both are evaluated with the same RNEA call, so there is little reason not to use the model when you have one.

## Operational space control

Policies often command the hand, not the joints. Khatib's **operational space** formulation writes the dynamics directly in hand coordinates $$x$$ [4]. Substituting $$\ddot x = J\ddot q + \dot J\dot q$$ into the manipulator equation and multiplying by $$J M^{-1}$$ gives the hand's apparent inertia

$$
\Lambda(q) = \big(J M^{-1} J^\top\big)^{-1}
\tag{8.12}\label{eq:lambda}
$$

and the control law

$$
\tau = J^\top \Big[\Lambda\big(\ddot x_{\text{ref}} + K_d \dot e_x + K_p e_x\big) + \mu + p\Big] + \big(I - J^\top \bar J^\top\big)\tau_0,
\qquad
\bar J = M^{-1}J^\top\Lambda ,
\tag{8.13}\label{eq:osc}
$$

where $$\mu$$ and $$p$$ are the Coriolis and gravity terms seen at the hand. $$\bar J$$ is the **dynamically consistent** pseudo inverse. The projector $$I - J^\top \bar J^\top$$ lets a secondary torque $$\tau_0$$, for example posture control towards $$q_{\text{nom}}$$, act without producing any hand acceleration. It is the dynamic counterpart of the kinematic null space of Section 7.2, weighted by the mass matrix. The ETH notes derive the same law and extend it to several prioritised tasks [2].

Equation \eqref{eq:lambda} inherits the singularities of Chapter 7. As $$J$$ loses rank, $$\Lambda$$ grows without bound along the lost direction. Implementations damp the inverse, exactly as in Section 7.2, or switch to joint space near singularities.

Setting $$\ddot x_{\text{ref}} = 0$$ and choosing $$K_p, K_d$$ as a desired stiffness and damping turns equation \eqref{eq:osc} into Hogan's **impedance control** [9]. The hand then behaves like a spring and damper around its target. It is the usual low level controller for contact rich manipulation. A common deployment pattern for the policies of this book is to send hand targets at the policy rate to an impedance controller running on the arm at 1 kHz.

{: .warning }
> Equations \eqref{eq:computed-torque} and \eqref{eq:osc} can request torques beyond what the motors provide. Clipping the torque destroys the cancellation and the stability argument with it. Keep the reference within the arm's dynamic limits, or treat limits explicitly with model predictive control.

{: .exercise }
> Run `simulate` with `tau_max = np.array([320, 320, 176, 176, 110, 40, 40])`, roughly the iiwa 14 limits, and raise `FREQ` until the computed torque controller saturates. Compare how quickly the error grows for computed torque and for PD with gravity compensation once the limits are active.

## References

1. R. Tedrake. *Robotic Manipulation.* Chapter "Force Control". [manipulation.mit.edu/force.html](https://manipulation.mit.edu/force.html)
2. Robotic Systems Lab, ETH Zurich. *Robot Dynamics Lecture Notes.* HS 2017, Sections 3.8 and 3.9 on joint space and task space dynamics control. [PDF](https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2017/RD_HS2017script.pdf)
3. R. Tedrake. *Underactuated Robotics.* Chapter "Acrobots, Cart-Poles, and Quadrotors", section on partial feedback linearisation. [underactuated.mit.edu](https://underactuated.mit.edu)
4. O. Khatib. *A Unified Approach for Motion and Force Control of Robot Manipulators: The Operational Space Formulation.* IEEE Journal on Robotics and Automation 3(1), 1987.
5. M. Takegaki, S. Arimoto. *A New Feedback Method for Dynamic Control of Manipulators.* Journal of Dynamic Systems, Measurement, and Control 103(2), 1981.
6. H. K. Khalil. *Nonlinear Systems.* 3rd edition, Prentice Hall, 2002. Section 4.2.
7. J.-J. E. Slotine, W. Li. *On the Adaptive Control of Robot Manipulators.* International Journal of Robotics Research 6(3), 1987.
8. C. G. Atkeson, C. H. An, J. M. Hollerbach. *Estimation of Inertial Parameters of Manipulator Loads and Links.* International Journal of Robotics Research 5(3), 1986.
9. N. Hogan. *Impedance Control: An Approach to Manipulation.* Journal of Dynamic Systems, Measurement, and Control 107(1), 1985.
10. G. Zardini, J. Lorenzetti, M. Pavone. *Principles of Robot Autonomy.* Stanford lecture notes, 2024. Sections on trajectory tracking and closed loop control. [PDF](https://stanfordasl.github.io/PoRA-I/aa174a_aut2425/pdfs/AA_PoRA_lecture_notes_2425.pdf)
