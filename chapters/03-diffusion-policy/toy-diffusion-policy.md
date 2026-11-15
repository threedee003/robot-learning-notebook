---
title: "3.1 A Minimal Diffusion Policy"
parent: "3. Diffusion Policy"
nav_order: 1
---

# 3.1 A Minimal Diffusion Policy
{: .no_toc }

1. TOC
{:toc}

## Setup

We return to the fork task from Section 1.1 and change one thing. Instead of regressing one displacement, the policy samples a chunk of $$T_p = 8$$ displacements, conditioned on the current position, and executes the first $$T_a = 4$$. The denoiser is the MLP from Section 2.1 with `x_dim = 16` and `cond_dim = 2`.

{% include codefile.html file="ch03_diffusion_policy_toy.py" %}

`chunk_dataset` slides a window over every demonstration. Near the end of a demonstration the window runs past the last action, so it is padded with zero displacements. That teaches the policy to stay at the goal.

## Results

```text
ddpm: rollouts {'success': 47, 'collision': 3, 'timeout': 0}  fork samples left 280 middle 21 right 199
ddim: rollouts {'success': 43, 'collision': 7, 'timeout': 0}  fork samples left 300 middle 71 right 129
```

{% include figure.html src="/assets/media/ch03/dp_toy_rollouts.png" alt="Rollouts of the diffusion policy with DDPM and DDIM samplers" caption="Fifty rollouts per sampler. Blue went left, orange went right, red failed." %}

Compare with Section 1.1. The policy now goes around the obstacle on **both** sides, which no regression policy can do from the same start. The fork statistics come from 500 chunks sampled at the fork point and sorted by their total sideways motion over the executed part. With DDPM only 21 of 500 chunks fall in the middle band, close to straight ahead. With 10 step DDIM 71 do. That is the deterministic sampler effect described at the end of Section 2.2.

{: .note }
> With 50 trials, 47 against 43 successes is not a clear separation. The 95% Wilson intervals are $$[0.84, 0.98]$$ and $$[0.74, 0.93]$$, which overlap (Chapter 6 shows how to compute them). Before the evaluation was seeded, the **same trained model** scored 45 with DDPM and anywhere from 36 to 39 with DDIM, depending only on the random numbers drawn during the rollouts. The 500 fork samples are the stronger evidence, because they measure the property we care about directly.

## A lesson from building this page

The first version of this policy concatenated the raw 2D position with the noisy chunk and the step embedding, giving an 82 dimensional input of which only 2 numbers carried the observation. It trained to a low loss and sampled one broad hump of chunks centred on straight ahead. In other words it averaged the modes, just like MSE.

Giving the condition its own two layer embedding, as in `Denoiser.c_embed`, fixed it immediately. The general rule is that the condition must be able to change the network's computation strongly, not just shift one input slightly. FiLM, used in the real Diffusion Policy network of Section 3.2, does exactly that by scaling and shifting every feature channel.

{: .warning }
> A diffusion policy with weak conditioning can look healthy on the training loss and still behave like a regression policy. Always sample many chunks at a known decision point and look at the spread, as `fork_modes` does.

## Where the remaining failures come from

The failures are almost all collisions right after the fork. They happen when a middle chunk is sampled, or when the robot replans after four steps and the new chunk picks the other side. The second effect is the reason the real Diffusion Policy executes $$T_a = 8$$ of $$T_p = 16$$ and conditions on two past observations. Seeing the last two positions tells the policy which way it was already moving. Section 3.3 adds both.

## References

1. C. Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* RSS 2023. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)
2. E. Perez et al. *FiLM: Visual Reasoning with a General Conditioning Layer.* AAAI 2018. [arXiv:1709.07871](https://arxiv.org/abs/1709.07871)
