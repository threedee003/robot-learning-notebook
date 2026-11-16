---
title: "7.2 Inverse Kinematics"
parent: "7. Manipulator Kinematics"
nav_order: 2
---

# 7.2 Inverse Kinematics
{: .no_toc }

1. TOC
{:toc}

## The problem has zero, finitely many, or infinitely many answers

Inverse kinematics asks for joint angles that put the hand at a desired pose $$X_d$$. It is a system of 6 nonlinear equations in $$n$$ unknowns, $$X(q) = X_d$$.

- Outside the workspace there is no solution.
- A 6 joint arm has finitely many solutions, up to 16 for a general geometry. An industrial arm typically has 8, from the choices elbow up or down, shoulder left or right, and wrist flipped or not.
- A 7 joint arm has a one dimensional family of solutions for each reachable pose, the **self motion manifold**. You can feel it by holding your hand still on a table and swinging your elbow.

When the last three axes of a 6 joint arm meet in a point, a **spherical wrist**, Pieper showed that position and orientation decouple [7]. The wrist point depends only on the first three joints, which reduces the problem to polynomial equations with closed form solutions. Analytic solvers such as IKFast exploit this. Tedrake recommends thinking about inverse kinematics as optimisation instead, because real problems add joint limits, collision avoidance and other constraints that no closed form handles [1]. That is the route we take.

## Newton's method on the pose error

Define the error of the current pose as a 6 vector, the rotation error in exponential coordinates followed by the position error, both in the world frame.

$$
e(q) = \begin{bmatrix} \log\!\big(R_d\, R(q)^\top\big) \\ p_d - p(q) \end{bmatrix}
\tag{7.10}\label{eq:pose-error}
$$

To first order a small joint step $$\Delta q$$ changes the hand by the twist $$J(q)\,\Delta q$$, so we want $$J(q)\,\Delta q = e(q)$$. With $$n = 7$$ and 6 equations this is underdetermined. The **pseudo inverse** $$J^+ = J^\top (JJ^\top)^{-1}$$ picks the smallest step that solves it, and iterating $$q \leftarrow q + J^+ e$$ is Newton's method. Near a solution where $$J$$ has full rank it converges quadratically [2, 8].

### Damping

Near a singularity a singular value $$\sigma_i$$ is tiny, and the pseudo inverse multiplies the error component along that direction by $$1/\sigma_i$$. The step explodes. The **damped least squares** step of Nakamura and Hanafusa [3] and Wampler [4] trades exactness for a bounded step by solving

$$
\Delta q = \arg\min_{\Delta q}\; \lVert J \Delta q - e \rVert^2 + \lambda^2 \lVert \Delta q \rVert^2
= J^\top \big(JJ^\top + \lambda^2 I\big)^{-1} e .
\tag{7.11}\label{eq:dls}
$$

In the singular value basis, the gain along direction $$i$$ changes from $$1/\sigma_i$$ to $$\sigma_i / (\sigma_i^2 + \lambda^2)$$. For $$\sigma_i \gg \lambda$$ this is the pseudo inverse. For $$\sigma_i \ll \lambda$$ it goes to zero instead of infinity, and its maximum value is $$1/(2\lambda)$$. Buss compares this step with the pseudo inverse and the Jacobian transpose on the same problems [6].

A constant $$\lambda$$ has a cost. Near the solution the error is small and the damping is no longer needed, but it still shrinks the step along weak directions, so convergence slows to linear. Sugihara's **Levenberg Marquardt** variant sets the damping from the error itself, $$\lambda^2 = e^\top e + \bar\lambda$$ with a small $$\bar\lambda$$ [5]. Far from the target the damping is large and the steps are cautious. Near the target it vanishes and Newton's quadratic rate returns.

### The null space

For a redundant arm, any joint velocity in the null space of $$J$$ moves the elbow without moving the hand. The projector onto that space is $$P = I - J^+ J$$. Adding $$P z$$ to any step leaves the first order hand motion unchanged, so a secondary goal can ride along for free. A common choice is $$z = k\,(q_{\text{nom}} - q)$$, which pulls the arm towards a comfortable posture and away from joint limits.

## Implementation

`ch07_ik.py` implements the four variants, a teleoperation loop for the next section, and the experiments.

{% include codefile.html file="ch07_ik.py" %}

## Results on random targets

The targets are 300 hand poses produced by forward kinematics from random joint angles within 90% of the limits, so every target is reachable. Every solve starts from the same nominal posture $$q_{\text{nom}} = (0, 30, 0, -60, 0, 90, 0)$$ degrees, takes at most 200 iterations, and succeeds when the position error is below 0.1 mm and the rotation error below $$10^{-3}$$ rad.

```text
method      success  median iters  mean |q - q_nom|
pinv         255/300             8             3.518
dls          255/300             9             3.350
lm           280/300            14             3.020
dls+null     249/300            12             3.133
near-singular target, pinv    : converged True  iterations 9
near-singular target, dls     : converged True  iterations 110
near-singular target, lm      : converged True  iterations 14
lm with up to 4 random restarts: 300/300
```

Three observations.

1. On random targets the pseudo inverse and fixed damping are almost indistinguishable. Random poses are rarely near a singularity, so the damping seldom matters. The failures come from joint limits. Clipping a step at a limit can leave the solver stuck in a corner of joint space from which the target is not reachable without first moving away.
2. Levenberg Marquardt solves 25 more targets. Its large early damping produces short steps, which rarely slam into limits, and it still converges in 14 iterations.
3. Null space centering found solutions closer to the nominal posture, but slightly fewer of them. A single start point is the real bottleneck. With up to four random restarts every target is solved.

{% include figure.html src="/assets/media/ch07/ik_convergence.png" alt="Error against iteration for three IK methods on a near singular target" caption="A target with the elbow at 1 degree, close to the straight elbow singularity. Fixed damping stalls in the weak direction. Error dependent damping converges almost as fast as the undamped Newton step." width="70%" %}

{: .key }
> For offline inverse kinematics, use error dependent damping and random restarts. Fixed damping buys robustness at the price of slow convergence exactly where the arm is most constrained.

## Differential inverse kinematics as a controller

Teleoperation and many learned policies command a hand **velocity** $$\mathcal{V}_d$$ instead of a pose. The controller must turn it into joint velocities at every control tick. The obvious choice is $$v = J^+ \mathcal{V}_d$$. Tedrake shows why this is dangerous and replaces it with a small optimisation that respects the robot's limits [1].

$$
\max_{v,\;\alpha}\; \alpha
\quad\text{subject to}\quad
J(q)\,v = \alpha\,\mathcal{V}_d,\quad
0 \le \alpha \le 1,\quad
v_{\min} \le v \le v_{\max},\quad
q_{\min} \le q + h\,v \le q_{\max}
\tag{7.12}\label{eq:diffik-qp}
$$

The scale factor $$\alpha$$ keeps the direction of the commanded motion and gives up its speed when the limits cannot be met. $$v = 0, \alpha = 0$$ is always feasible, so the robot stops instead of doing something unexpected. Our version adds a small penalty $$\tfrac{\varepsilon}{2}\lVert v \rVert^2$$ to pick a unique minimum norm solution and solves it with SciPy's SLSQP. A production controller would use a dedicated QP solver, which is what Drake does.

The experiment commands the hand straight up at 10 cm/s, holding its orientation, from a bent posture. Straight up leads into the elbow singularity of Section 7.1, which the arm reaches when the hand is at about 0.91 m.

```text
pinv final z 1.007 m  max |v|/v_max  178.63  limit first exceeded  0.93 s  largest hand jump in one step  49.85 cm  final alpha 1.00
qp   final z 0.918 m  max |v|/v_max    1.00  limit first exceeded   never  largest hand jump in one step   0.10 cm  final alpha 0.02
```

{% include figure.html src="/assets/media/ch07/diffik_singularity.png" alt="Hand height and joint speed ratio for pseudo inverse and QP differential IK" caption="Left, hand height. Right, the largest joint speed as a fraction of its limit. The pseudo inverse controller exceeds the limits by a factor of 179, flips the elbow through the singularity in a single 10 ms step and drops the hand by half a metre. The QP controller slows down and stops at the singularity." %}

The pseudo inverse run ends higher, which looks like success in a plot of final positions. It got there by teleporting the hand by 50 cm in 10 ms. On hardware the drive would fault on a velocity limit, or the arm would hit something. The QP run stops at the edge of the workspace for this orientation, with $$\alpha = 0.02$$ signalling that the command cannot be followed.

{: .warning }
> The orientation part of $$e$$ is in radians and the position part in metres. Their relative weight is arbitrary, and one radian of error is not equivalent to one metre. If your solver converges in position but not orientation, or the reverse, add a diagonal weight to both terms of equation \eqref{eq:dls} and report it.

{: .exercise }
> Set `eps` in `diff_ik_qp` to zero. The problem becomes a linear program, and the solver returns a vertex of the feasible set. Plot the joint velocities. Why do several joints now sit exactly at their limits even far from the singularity, and why is that bad for a real robot?

## References

1. R. Tedrake. *Robotic Manipulation.* Chapters "Basic Pick and Place" (differential inverse kinematics with constraints) and "Motion Planning" (inverse kinematics as optimisation). [manipulation.mit.edu](https://manipulation.mit.edu/pick.html)
2. Robotic Systems Lab, ETH Zurich. *Robot Dynamics Lecture Notes.* HS 2017, Section 2.9 on kinematic control and inverse differential kinematics. [PDF](https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2017/RD_HS2017script.pdf)
3. Y. Nakamura, H. Hanafusa. *Inverse Kinematic Solutions with Singularity Robustness for Robot Manipulator Control.* Journal of Dynamic Systems, Measurement, and Control 108(3), 1986.
4. C. W. Wampler. *Manipulator Inverse Kinematic Solutions Based on Vector Formulations and Damped Least-Squares Methods.* IEEE Transactions on Systems, Man, and Cybernetics 16(1), 1986.
5. T. Sugihara. *Solvability-Unconcerned Inverse Kinematics by the Levenberg-Marquardt Method.* IEEE Transactions on Robotics 27(5), 2011.
6. S. R. Buss. *Introduction to Inverse Kinematics with Jacobian Transpose, Pseudoinverse and Damped Least Squares Methods.* Technical report, University of California San Diego, 2004.
7. D. L. Pieper. *The Kinematics of Manipulators under Computer Control.* PhD thesis, Stanford University, 1968.
8. K. M. Lynch, F. C. Park. *Modern Robotics.* Cambridge University Press, 2017. Chapter 6. [modernrobotics.org](http://modernrobotics.org)
