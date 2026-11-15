---
title: "4.2 Straighter Paths"
parent: "4. Flow Matching"
nav_order: 2
---

# 4.2 Straighter Paths: OT Coupling and Rectified Flow
{: .no_toc }

1. TOC
{:toc}

## Where the curvature comes from

Section 4.1 showed that the learned flow is curved even though every training path is a straight line. The reason is the **coupling**. During training each data point $$\mathbf{x}_1$$ is paired with an independent noise sample $$\mathbf{x}_0$$. A noise sample on the left is as likely to be paired with a data point on the right as on the left. The straight lines for different pairs cross, and at each crossing the marginal field averages the directions. Averaged directions bend the trajectories.

If the pairs did not cross, the field would not need to average, and the trajectories would stay straight. Two practical ways to get closer to that are below.

## Minibatch optimal transport

Within each training batch of $$B$$ noise samples and $$B$$ data points, re-pair them to minimise the total squared distance.

$$
\sigma^\star = \arg\min_{\sigma \in S_B} \sum_{i=1}^{B} \big\lVert \mathbf{x}_0^{(i)} - \mathbf{x}_1^{(\sigma(i))} \big\rVert^2
$$

This is a linear assignment problem, solved exactly by the Hungarian algorithm in `scipy`. Pairs from an optimal assignment never cross, since swapping the endpoints of two crossing pairs would always shorten the total. As $$B$$ grows the coupling approaches the optimal transport plan between the two distributions. Tong et al. and Pooladian et al. proposed this independently [1, 2].

{% include codefile.html file="ch04_ot_cfm.py" %}

The training loop of Section 4.1 accepts a `coupling` function, so OT flow matching is one argument.

## Results

```text
independent  euler  1 steps  on-mode 0.000  modes 0/8
independent  euler  2 steps  on-mode 0.179  modes 8/8
independent  euler  4 steps  on-mode 0.775  modes 8/8
independent  euler 10 steps  on-mode 0.936  modes 8/8
independent  euler 50 steps  on-mode 0.969  modes 8/8
independent  path length / displacement = 1.438
minibatch-OT euler  1 steps  on-mode 0.781  modes 8/8
minibatch-OT euler  2 steps  on-mode 0.878  modes 8/8
minibatch-OT euler  4 steps  on-mode 0.919  modes 8/8
minibatch-OT euler 10 steps  on-mode 0.941  modes 8/8
minibatch-OT euler 50 steps  on-mode 0.950  modes 8/8
minibatch-OT path length / displacement = 1.008
```

{% include figure.html src="/assets/media/ch04/flow_paths.png" alt="Sampled trajectories from noise to the eight Gaussians for independent and OT coupling" caption="Trajectories of 300 samples, 50 Euler steps. Grey dots are noise, orange dots are samples." %}

The paths with OT coupling are almost exactly straight, with a length ratio of 1.008 against 1.438. The effect on few step sampling is large. One Euler step goes from 0% to 78% of samples on a mode, and two steps from 18% to 88%. At 50 steps the independent model is slightly better (0.969 against 0.950). A batch of 512 only approximates the true optimal transport plan, so the OT model has learned a slightly biased field, which costs a little accuracy once integration error no longer dominates.

## Rectified flow

Liu et al. take a different route [3]. Train a flow with any coupling. Then generate new pairs by running the trained flow, $$(\mathbf{x}_0, \mathbf{x}_1 = \mathrm{ODESolve}(\mathbf{x}_0))$$, and train a second flow on these pairs. The new pairs come from a deterministic flow whose paths never cross, so the second flow is straighter. This step is called **reflow** and can be repeated. `reflow_pairs` in the file above generates such pairs. Each round costs a full training run, but the result can be sampled in one or two steps.

| Method | Extra training cost | Effect on paths |
|:--|:--|:--|
| Independent coupling | none | curved where data are ambiguous |
| Minibatch OT | one assignment per batch, $$O(B^3)$$ | straighter, gain grows with batch size |
| Reflow | one more training run per round | straighter after each round |
| Distillation (consistency models) | one more training run | one step sampling |

## Does this matter for policies?

For conditional generation the coupling problem is smaller. The observation already narrows the target distribution, so most of the ambiguity that bends paths is gone before any coupling is applied (Chapter 5). Minibatch OT is also awkward with conditioning, since the assignment should only pair samples that share an observation. In practice, policies such as pi0 use independent coupling and rely on few step Euler integration [4]. When latency is critical, distillation into one step models is the more common route [5].

## References

1. A. Tong et al. *Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport.* TMLR 2024. [arXiv:2302.00482](https://arxiv.org/abs/2302.00482)
2. A.-A. Pooladian et al. *Multisample Flow Matching: Straightening Flows with Minibatch Couplings.* ICML 2023. [arXiv:2304.14772](https://arxiv.org/abs/2304.14772)
3. X. Liu, C. Gong, Q. Liu. *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.* ICLR 2023. [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
4. K. Black et al. *pi0: A Vision-Language-Action Flow Model for General Robot Control.* 2024. [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)
5. A. Prasad et al. *Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation.* RSS 2024. [arXiv:2405.07503](https://arxiv.org/abs/2405.07503)
