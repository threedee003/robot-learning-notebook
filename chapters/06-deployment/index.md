---
title: "6. From Simulation to Robot"
nav_order: 7
has_children: true
---

# 6. From Simulation to Robot
{: .no_toc }

<p class="rlb-meta">Prerequisites: Chapters 3 and 5</p>

1. TOC
{:toc}

## The control loop has a clock

In simulation the world waits while the policy thinks. On hardware it does not. If inference takes 80 ms and the controller runs at 10 Hz, the robot has moved for most of a cycle before the first action of the new chunk is ready. Two consequences follow.

1. The first actions of a fresh chunk are already in the past when they arrive.
2. Switching from the old chunk to the new one can cause a jump if the two disagree.

The fix for the first problem is to **timestamp** every action. Give action $$i$$ of a chunk predicted from an observation captured at time $$t_{\text{obs}}$$ the target time $$t_{\text{obs}} + i\,\Delta t$$. When the chunk arrives at time $$t_{\text{now}}$$, drop every action with a target time earlier than $$t_{\text{now}}$$ plus the execution latency of the controller. The real robot code of Diffusion Policy does this [1]. It requires observations and actions to be logged with timestamps from the same clock during data collection too.

The fix for the second problem is either a longer action horizon $$T_a$$, which makes switches rarer, or **temporal ensembling**, which blends overlapping chunks (Section 6.1).

## Latency budget

| Stage | Typical cost | How to reduce |
|:--|:--|:--|
| Camera capture and transfer | 10 to 40 ms | Lower resolution, hardware sync |
| Image encoder | 2 to 10 ms on GPU | Run once per control step, not per denoising step |
| Denoising loop | $$N \times$$ network cost | Fewer steps (flow matching, DDIM), distillation [2] |
| Controller and network to robot | 1 to 10 ms | Real time kernel, wired Ethernet |

Measure each stage separately on the target computer. The order of magnitude changes between a laptop CPU and an embedded GPU, and the denoising loop is the only part this book's choices directly control.

## Action representation

| Choice | Recommendation | Reason |
|:--|:--|:--|
| Position or velocity | Absolute end effector or joint positions | Less compounding error, works with multimodal models [1] |
| Rotations | 6D representation (two columns of the rotation matrix) | Continuous, unlike Euler angles and quaternions [3] |
| Gripper | One scalar, thresholded at execution | Keeps the chunk a smooth time series |
| Normalisation | Min max to $$[-1, 1]$$ per dimension | Matches the clip range of the samplers |

{: .warning }
> A dimension that is constant in the training data, for example a joint that never moved, has $$\max - \min = 0$$. Min max normalisation then divides by zero or by $$10^{-8}$$, and any tiny deviation at test time becomes a huge input. Detect such dimensions and set their range to 1.

## Evaluating honestly

A success rate from a handful of trials says little. With $$n$$ trials and $$s$$ successes, the Wilson score interval gives a range that is reliable even for small $$n$$ and rates near 0 or 1 [4].

$$
\frac{\hat p + \frac{z^2}{2n}}{1 + \frac{z^2}{n}} \;\pm\; \frac{z}{1 + \frac{z^2}{n}}\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}},
\qquad \hat p = s/n,\; z = 1.96
$$

{% include codefile.html file="ch06_eval.py" %}

```text
45/50  rate 0.90  95% interval [0.79, 0.96]
39/50  rate 0.78  95% interval [0.65, 0.87]
 9/10  rate 0.90  95% interval [0.60, 0.98]
18/20  rate 0.90  95% interval [0.70, 0.97]
```

Nine out of ten is compatible with a true rate of 60%. When comparing two policies, run both on the same set of initial conditions, alternate between them so that lighting and wear affect both equally, and report the intervals.

## Debugging checklist

| Symptom | Likely cause | Check |
|:--|:--|:--|
| Loss falls, robot does nothing sensible | Normaliser fitted on different data than used at test time | Print normalised observations at test time, they should lie in $$[-1, 1]$$ |
| Policy ignores the observation | Weak conditioning | Sample many chunks for two very different observations and compare |
| Jerky motion at chunk boundaries | Mode switching between chunks | Increase $$T_a$$ or use temporal ensembling |
| Actions blow up with DDIM | Missing $$\hat{\mathbf{x}}_0$$ clipping | Section 2.2 |
| Flow policy moves away from targets | Time convention flipped | Section 5, warning box |
| Good in sim, fails on hardware | Latency not compensated | Timestamp actions, log inference time |
| EMA model worse than raw model | BatchNorm statistics | Use GroupNorm, Section 3.2 |

## References

1. C. Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* RSS 2023. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137). Real robot code: [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy)
2. A. Prasad, K. Lin, J. Wu, L. Zhou, J. Bohg. *Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation.* RSS 2024. [arXiv:2405.07503](https://arxiv.org/abs/2405.07503)
3. Y. Zhou, C. Barnes, J. Lu, J. Yang, H. Li. *On the Continuity of Rotation Representations in Neural Networks.* CVPR 2019. [arXiv:1812.07035](https://arxiv.org/abs/1812.07035)
4. E. B. Wilson. *Probable Inference, the Law of Succession, and Statistical Inference.* JASA 22(158), 1927.
