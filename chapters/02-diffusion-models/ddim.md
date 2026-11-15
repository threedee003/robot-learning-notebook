---
title: "2.2 Faster Sampling with DDIM"
parent: "2. Diffusion Models"
nav_order: 2
---

# 2.2 Faster Sampling with DDIM
{: .no_toc }

1. TOC
{:toc}

## The cost of 100 steps

DDPM sampling calls the network once per diffusion step. For a robot that replans at 10 Hz with a large network, 100 calls per chunk is too slow. Song, Meng and Ermon observed that the training loss only depends on the marginals $$q(\mathbf{x}_k \mid \mathbf{x}_0)$$, not on the Markov chain that produced them [1]. Any other process with the same marginals can reuse the trained network. Some of those processes are deterministic and can skip steps.

## Deriving the update

Given $$\mathbf{x}_k$$ and the predicted noise $$\hat{\boldsymbol\epsilon} = \boldsymbol\epsilon_\theta(\mathbf{x}_k, k)$$, invert equation (2.1) to estimate the clean sample.

$$
\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_k - \sqrt{1-\bar\alpha_k}\,\hat{\boldsymbol\epsilon}}{\sqrt{\bar\alpha_k}}
\tag{2.4}\label{eq:x0hat}
$$

Now jump to any earlier step $$k' < k$$ by re-noising $$\hat{\mathbf{x}}_0$$ with the **same** noise estimate instead of fresh noise.

$$
\mathbf{x}_{k'} = \sqrt{\bar\alpha_{k'}}\,\hat{\mathbf{x}}_0 + \sqrt{1-\bar\alpha_{k'}}\,\hat{\boldsymbol\epsilon}
\tag{2.5}
$$

If the network were perfect, this would place $$\mathbf{x}_{k'}$$ exactly on the marginal $$q(\mathbf{x}_{k'} \mid \mathbf{x}_0)$$. The update has no random term. It is a first order discretisation of the probability flow ODE of the diffusion [2], so the gap between $$k$$ and $$k'$$ can be large. With 10 sampling steps we visit $$k = 99, 88, 77, \dots, 0$$. The general DDIM family adds a parameter $$\eta$$ that blends in fresh noise. $$\eta = 1$$ recovers DDPM and $$\eta = 0$$ is the deterministic version used here.

## A numerical trap

Look at the denominator of equation \eqref{eq:x0hat}. With the cosine schedule and $$K = 100$$, the first step has

$$
\bar\alpha_{99} \approx 2.4 \times 10^{-7}, \qquad \frac{1}{\sqrt{\bar\alpha_{99}}} \approx 2000.
$$

Any error in $$\hat{\boldsymbol\epsilon}$$ is multiplied by about 2000 in $$\hat{\mathbf{x}}_0$$. DDPM never forms $$\hat{\mathbf{x}}_0$$ explicitly, so it is unaffected. DDIM carries the inflated estimate into the next step. The standard remedy is to clip $$\hat{\mathbf{x}}_0$$ to the known range of the data and then recompute $$\hat{\boldsymbol\epsilon}$$ from the clipped value, so that equation (2.5) stays consistent.

```python
x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
x0 = x0.clamp(-clip, clip)
eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
```

Diffusion Policy normalises actions to $$[-1, 1]$$ and clips to the same range, which is why `clip=1.0` appears in Chapter 3. For the ring data, which spans about $$[-2.3, 2.3]$$, we clip at 3.

## Results

The same trained model as Section 2.1, sampled with DDIM.

```text
DDIM   5 steps  on-mode  no clip 0.193   clip 0.859
DDIM  10 steps  on-mode  no clip 0.608   clip 0.919
DDIM  25 steps  on-mode  no clip 0.553   clip 0.936
DDIM  50 steps  on-mode  no clip 0.517   clip 0.941
```

Without clipping, quality peaks at 10 steps and then gets **worse** with more steps, which should never happen for a correct ODE solver. Every extra step near pure noise is another division by a tiny $$\sqrt{\bar\alpha_k}$$. With clipping, quality rises steadily with the number of steps, and 10 steps reach 0.92 against 0.98 for DDPM with 100. This was found while writing this chapter. The first version of the code had no clipping and produced the left column.

{: .key }
> Fewer sampling steps is the single biggest lever on policy latency. DDIM gives it for free on a trained DDPM model, provided $$\hat{\mathbf{x}}_0$$ is clipped. Chapter 4 shows a formulation, flow matching, that avoids the problem by construction.

## Deterministic samplers and modes

There is a second, subtler effect. On the 2D ring every sampler covers all eight modes, but in Section 3.1 we will see that 10 step DDIM puts more samples **between** the two modes of the fork task than stochastic DDPM does. The injected noise in DDPM corrects earlier errors of the network. A deterministic sampler follows the learned field exactly, errors included. Karras et al. analyse this trade off in detail [3].

## References

1. J. Song, C. Meng, S. Ermon. *Denoising Diffusion Implicit Models.* ICLR 2021. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)
2. Y. Song et al. *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR 2021. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
3. T. Karras, M. Aittala, T. Aila, S. Laine. *Elucidating the Design Space of Diffusion-Based Generative Models.* NeurIPS 2022. [arXiv:2206.00364](https://arxiv.org/abs/2206.00364)
