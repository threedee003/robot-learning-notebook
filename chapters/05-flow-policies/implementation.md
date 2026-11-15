---
title: "5.1 Implementing a Flow Policy"
parent: "5. Flow Matching Policies"
nav_order: 1
---

# 5.1 Implementing a Flow Policy
{: .no_toc }

1. TOC
{:toc}

## What we reuse

We reuse everything from Section 3.3 except the generative objective. That means the dataset windows, the normaliser, the EMA, the U-Net and the episode runner. The U-Net takes a "step" input that was an integer in $$\{0, \dots, 99\}$$. For flow matching we pass $$1000\,\tau$$ instead. The sinusoidal embedding accepts any real number, and the scaling keeps its frequencies in a useful range.

## The flow matcher

{% include codefile.html file="flow_core.py" %}

`training_loss` is equation (4.3) with an optional Beta time distribution. `sample` integrates the ODE with Euler or the midpoint rule. Compare with `NoiseSchedule` in Section 2.1. There are no schedule tensors, no posterior variance and no clipping step.

## The policy

{% include codefile.html file="ch05_flow_policy.py" %}

## Results

```text
iter     0  loss 1.7782
iter   500  loss 0.0466
iter  1000  loss 0.0280
iter  1500  loss 0.0468
iter  2000  loss 0.0201
iter  2500  loss 0.0273
flow policy  1 Euler steps: {'collision': 13, 'success': 37}
flow policy  3 Euler steps: {'success': 50}
flow policy 10 Euler steps: {'success': 50}
```

Three Euler steps are enough for 50 of 50 on this task. One step fails 13 times, and every failure is a collision with the obstacle rather than a timeout. That fits the prediction of the chapter introduction. Away from the fork the conditional distribution is narrow, the flow is nearly straight, and one step is exact. At the fork the field must bend to send noise to two different modes, and a single straight jump lands between them.

{: .note }
> The flow loss is larger than the diffusion loss of Section 3.3 (about 0.02 against 0.01). The two numbers are not comparable. The flow target $$\mathbf{A} - \boldsymbol\epsilon$$ has the variance of the data plus 1 in every dimension, while the noise target $$\boldsymbol\epsilon$$ has variance 1, and the two losses weight the noise levels differently.

## Latency

Success is only half the story. The cost of a chunk is the number of network calls times the cost of one call. A separate benchmark measures that cleanly on one CPU thread.

{% include codefile.html file="ch05_latency.py" %}

```text
toy U-Net (32, 64, 128), 2-D actions
   DDPM 100 steps          237.0 ms per chunk
   DDIM 10 steps            26.1 ms per chunk
   flow 10 Euler steps      25.0 ms per chunk
   flow 3 Euler steps        7.1 ms per chunk
   flow 1 Euler step         2.3 ms per chunk
paper U-Net (256, 512, 1024), 7-D actions
   DDPM 100 steps         4909.7 ms per chunk
   DDIM 10 steps           489.4 ms per chunk
   flow 10 Euler steps     491.7 ms per chunk
   flow 3 Euler steps      147.2 ms per chunk
   flow 1 Euler step        47.6 ms per chunk
```

These were measured on one thread of a cloud CPU, so treat the absolute values as an upper bound for a laptop and far above a GPU. The cost is linear in the number of network calls, as expected. For the paper sized network, 100 DDPM steps take about 5 seconds per chunk on a CPU, far too slow for a 10 Hz loop. Ten steps of either DDIM or flow matching cost the same, since both make ten calls. The advantage of flow matching is that its quality at few steps holds up without tricks such as clipping, and that 3 steps were enough here.

{: .exercise }
> Train with `time_sampling="uniform"` instead of `"beta"` and compare the 1 step and 3 step success rates. Then try `method="midpoint"` with 2 steps, which also costs 4 network calls. Is it better than 4 Euler steps?

## References

1. Y. Lipman et al. *Flow Matching for Generative Modeling.* ICLR 2023. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
2. K. Black et al. *pi0: A Vision-Language-Action Flow Model for General Robot Control.* 2024. [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)
