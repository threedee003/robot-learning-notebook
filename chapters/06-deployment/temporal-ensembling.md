---
title: "6.1 Temporal Ensembling"
parent: "6. From Simulation to Robot"
nav_order: 1
---

# 6.1 Temporal Ensembling
{: .no_toc }

1. TOC
{:toc}

## Overlapping chunks

Receding horizon control throws away most of every chunk. Temporal ensembling, introduced with ACT [1], uses all of it. The policy is queried at **every** control step. At time $$t$$ several earlier chunks contain a prediction for the action at $$t$$. Call them $$\mathbf{a}_t^{(0)}, \dots, \mathbf{a}_t^{(m)}$$, with index 0 the oldest. The executed action is a weighted average.

$$
\mathbf{a}_t = \frac{\sum_i w_i\, \mathbf{a}_t^{(i)}}{\sum_i w_i}, \qquad w_i = \exp(-m\, i)
$$

With $$m > 0$$ the oldest prediction gets the largest weight, which favours smoothness. A smaller $$m$$ lets new observations take effect faster. The ACT paper uses this weighting with the oldest prediction weighted most [1].

## Implementation

{% include codefile.html file="ch06_temporal_ensemble.py" %}

The buffer row $$s$$ holds the chunk predicted at step $$s$$, written into columns $$s$$ to $$s + T_p - 1$$. Reading column $$t$$ collects every prediction for time $$t$$. The self test feeds noisy chunks whose correct value at time $$t$$ is $$t$$.

```text
t=0  executed 0.038  target 0
t=1  executed 0.915  target 1
t=2  executed 2.091  target 2
t=3  executed 2.988  target 3
t=4  executed 4.050  target 4
t=5  executed 4.883  target 5
```

The noise on each chunk has standard deviation 0.3. After a few steps up to four predictions are averaged and the executed error drops well below that.

## Ensembling and multimodality

{: .warning }
> Averaging is the operation that broke MSE behaviour cloning in Chapter 1. If two overlapping chunks come from **different** modes, the ensemble averages them and drives the robot into the gap between them. Temporal ensembling suits deterministic or nearly unimodal policies such as ACT. With a diffusion or flow policy at a genuine decision point, prefer a longer action horizon $$T_a$$ so the robot commits to one sample.

A simple safeguard is to check the spread of the predictions before averaging. If the distance between the oldest and the newest prediction exceeds a threshold, execute the oldest one alone instead of the mean.

{: .exercise }
> Use the flow policy from Section 5.1 with `ACTION_HORIZON = 1` and wrap it with `TemporalEnsembler`. Measure the success rate on the fork task for $$m \in \{0, 0.1, 1\}$$. Which value of $$m$$ comes closest to always executing the oldest prediction?

## References

1. T. Z. Zhao, V. Kumar, S. Levine, C. Finn. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.* RSS 2023. [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)
