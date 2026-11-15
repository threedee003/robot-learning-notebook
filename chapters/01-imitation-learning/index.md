---
title: "1. Imitation Learning"
nav_order: 2
has_children: true
---

# 1. Imitation Learning
{: .no_toc }

<p class="rlb-meta">Prerequisites: maximum likelihood, basic PyTorch</p>

1. TOC
{:toc}

## The problem

A human teleoperates a robot and records what it saw and what it did. This gives a dataset of demonstrations

$$
\mathcal{D} = \{(\mathbf{o}_t^{(i)}, \mathbf{a}_t^{(i)})\}_{i=1..N,\; t=1..L_i}.
$$

Imitation learning asks for a policy $$\pi_\theta(\mathbf{a} \mid \mathbf{o})$$ that behaves like the demonstrator. There is no reward function. The demonstrations are the only specification of the task.

The simplest approach is **behaviour cloning** (BC). We treat the problem as supervised learning and maximise the likelihood of the demonstrated actions.

$$
\theta^\star = \arg\max_\theta \; \E_{(\mathbf{o}, \mathbf{a}) \sim \mathcal{D}} \left[ \log \pi_\theta(\mathbf{a} \mid \mathbf{o}) \right]
$$

## Why the policy class matters

Everything interesting in BC is hidden in the choice of $$\pi_\theta$$. The most common choice is a network $$f_\theta(\mathbf{o})$$ that outputs one action. This choice silently assumes a Gaussian with fixed variance.

$$
\pi_\theta(\mathbf{a} \mid \mathbf{o}) = \N\!\left(\mathbf{a};\, f_\theta(\mathbf{o}),\, \sigma^2 I\right)
\quad\Longrightarrow\quad
-\log \pi_\theta = \frac{1}{2\sigma^2} \lVert \mathbf{a} - f_\theta(\mathbf{o}) \rVert^2 + \text{const}
$$

So maximum likelihood under this model is exactly mean squared error. The minimiser of MSE over all functions is the conditional mean.

$$
f^\star(\mathbf{o}) = \E[\mathbf{a} \mid \mathbf{o}]
$$

{: .key }
> An MSE policy predicts the **average** of what the demonstrator did in a given situation. If the demonstrator sometimes went left and sometimes went right, the average is straight ahead. Straight ahead may be the one action nobody demonstrated.

This is the multimodality problem. Section 1.1 reproduces it on a toy task. It is the main reason the rest of this book uses generative models as policies.

## Compounding error

A second problem does not depend on the policy class. During training the policy only sees states from the demonstrator. During deployment it sees states produced by its own actions. A small mistake moves the robot slightly off the data, where the policy is less accurate, which causes a larger mistake.

Ross and Bagnell showed that if a BC policy errs with probability $$\epsilon$$ per step on the expert's state distribution, the expected cost over a horizon of $$T$$ steps can grow as $$O(\epsilon T^2)$$ [1]. Interactive methods such as DAgger reduce this to $$O(\epsilon T)$$ by querying the expert on the learner's own states [2]. Teleoperation data rarely allows that, so modern policies attack the problem differently.

**Action chunking** is the practical fix. The policy predicts a sequence of $$T_p$$ future actions and executes several of them before it replans. The number of independent decisions per episode falls by that factor, which shortens the effective horizon over which errors compound [3]. Chunking also helps with demonstrations that contain pauses or idle periods, where a single step policy cannot tell what phase of the task it is in.

## What a good policy class needs

| Requirement | Why it matters for robots |
|:--|:--|
| Multimodal | Demonstrators solve the same situation in different ways |
| High dimensional output | A chunk of 16 steps for a 7 DoF arm with gripper is 128 numbers |
| Temporally consistent | Neighbouring actions in a chunk must belong to the same mode |
| Stable to train | No adversarial games, no negative sampling |
| Fast enough to sample | Control loops run at 10 to 50 Hz |

The classical options each fail at least one row.

| Policy class | Multimodal | Scales to chunks | Notes |
|:--|:--:|:--:|:--|
| MSE regression | no | yes | Averages modes |
| Mixture density network [4] | partly | poorly | Number of modes fixed in advance, unstable at high dimension |
| Discretised actions, BeT [5] | yes | partly | Binning plus offsets, resolution limited |
| Implicit BC, energy model [6] | yes | poorly | Needs negative samples, training is unstable |
| **Diffusion Policy** [7] | yes | yes | Chapter 3 |
| **Flow matching policy** [8] | yes | yes | Chapter 5 |

## References

1. S. Ross, J. A. Bagnell. *Efficient Reductions for Imitation Learning.* AISTATS 2010. [PMLR](https://proceedings.mlr.press/v9/ross10a.html)
2. S. Ross, G. Gordon, J. A. Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.* AISTATS 2011. [arXiv:1011.0686](https://arxiv.org/abs/1011.0686)
3. T. Z. Zhao, V. Kumar, S. Levine, C. Finn. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.* RSS 2023. [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)
4. C. M. Bishop. *Mixture Density Networks.* Aston University technical report, 1994.
5. N. M. Shafiullah, Z. J. Cui, A. Altanzaya, L. Pinto. *Behavior Transformers: Cloning k Modes with One Stone.* NeurIPS 2022. [arXiv:2206.11251](https://arxiv.org/abs/2206.11251)
6. P. Florence et al. *Implicit Behavioral Cloning.* CoRL 2021. [arXiv:2109.00137](https://arxiv.org/abs/2109.00137)
7. C. Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* RSS 2023. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)
8. K. Black et al. *pi0: A Vision-Language-Action Flow Model for General Robot Control.* 2024. [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)
