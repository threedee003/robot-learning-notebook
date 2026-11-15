---
title: "4.1 Flow Matching from Scratch"
parent: "4. Flow Matching"
nav_order: 1
---

# 4.1 Flow Matching from Scratch
{: .no_toc }

1. TOC
{:toc}

## Same problem, new objective

We use the eight Gaussians and the MLP of Section 2.1 unchanged. The network output is now read as a velocity instead of a noise prediction. The `FlowMatcher` class implements equation (4.3) and an Euler or midpoint ODE solver.

{% include codefile.html file="flow_core.py" %}

Two details deserve a comment.

- **Time scaling.** The network receives $$1000\,\tau$$ instead of $$\tau$$. The sinusoidal embedding was designed for integer step indices up to about 1000. With raw $$\tau \in [0, 1]$$ the high frequency components barely change and the network has trouble telling nearby times apart.
- **No clipping.** Unlike DDIM, nothing here divides by a quantity that goes to zero. The Euler update only adds $$\Delta \cdot v_\theta$$.

## Training and evaluation

{% include codefile.html file="ch04_cfm_2d.py" %}

`straightness` divides the mean length of the sampled paths by the mean distance between their start and end points. A value of 1 means every path is a straight line, which is the case where one Euler step is exact.

## Results

```text
independent  euler  1 steps  on-mode 0.000  modes 0/8
independent  euler  2 steps  on-mode 0.179  modes 8/8
independent  euler  4 steps  on-mode 0.775  modes 8/8
independent  euler 10 steps  on-mode 0.936  modes 8/8
independent  euler 50 steps  on-mode 0.969  modes 8/8
independent  path length / displacement = 1.438
```

With one step, no sample lands on a mode. A single Euler step from noise moves every point by $$v_\theta(\mathbf{x}_0, 0)$$. At $$\tau = 0$$ the input is $$\mathbf{x}_0$$ itself, and under independent coupling $$\mathbf{x}_0$$ tells us nothing about $$\mathbf{x}_1$$. So the optimal field is $$v^\star(\mathbf{x}, 0) = \E[\mathbf{x}_1] - \mathbf{x}$$, and one step sends every sample to $$\mathbf{x} + v^\star = \E[\mathbf{x}_1]$$, the centre of the ring. The one step sampler collapses to the data mean, exactly like MSE regression. With 10 steps the model is close to its 50 step quality.

{: .key }
> The first Euler step of a flow model **is** a regression to the mean. Later steps undo that, because by then $$\mathbf{x}_\tau$$ carries information about the mode. The number of steps needed is set by how long the trajectories stay ambiguous.

{: .exercise }
> Compare `method="midpoint"` with 5 steps against Euler with 10 steps. Both cost 10 network calls. Which gives the higher on-mode fraction, and why does a second order method help most when paths are curved?

## References

1. Y. Lipman et al. *Flow Matching for Generative Modeling.* ICLR 2023. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
2. Y. Lipman et al. *Flow Matching Guide and Code.* 2024. [arXiv:2412.06264](https://arxiv.org/abs/2412.06264)
