---
title: "5.2 Diffusion or Flow?"
parent: "5. Flow Matching Policies"
nav_order: 2
---

# 5.2 Diffusion or Flow?
{: .no_toc }

1. TOC
{:toc}

## What this book measured

All numbers come from the scripts in this book, on the fork task with 50 rollouts each.

| Policy | Network | Sampler | Network calls | Success |
|:--|:--|:--|:--:|:--:|
| MSE regression (Section 1.1) | MLP | none | 1 | 50/50 or 3/50, depending on epochs |
| Diffusion (Section 3.1) | MLP | DDPM | 100 | 47/50 |
| Diffusion (Section 3.1) | MLP | DDIM | 10 | 43/50 |
| Diffusion (Section 3.3) | U-Net, $$T_o = 2$$ | DDIM | 10 | 46/50 |
| Flow (Section 5.1) | U-Net, $$T_o = 2$$ | Euler | 1 | 37/50 |
| Flow (Section 5.1) | U-Net, $$T_o = 2$$ | Euler | 3 | 50/50 |
| Flow (Section 5.1) | U-Net, $$T_o = 2$$ | Euler | 10 | 50/50 |

A toy task with 50 trials cannot rank methods for real robots. What it does show is the **mechanism** behind each result, and those mechanisms carry over.

## A side by side comparison

| Question | Diffusion (DDPM, DDIM) | Flow matching |
|:--|:--|:--|
| Training loss | MSE on noise | MSE on velocity |
| Hyperparameters of the generative part | $$K$$, $$\beta$$ schedule, clip range, $$\eta$$ | time distribution, number of steps |
| Few step sampling | needs DDIM and $$\hat{\mathbf{x}}_0$$ clipping | Euler works directly |
| Stochastic sampling | built in (DDPM), corrects model errors | possible via SDE variants, rarely used |
| Ecosystem | `diffusers` schedulers, Diffusion Policy reference code | `openpi`, `torchcfm`, simpler code |
| Large scale robot results | Diffusion Policy and many follow ups [1] | pi0 and successors [2] |

## Practical advice

1. **Start with whichever reference code is closest to your setup.** The data pipeline, normalisation and horizons matter more than the choice of generative objective. Both choices in this book share all of that code.
2. **If latency matters, prefer flow matching.** It reaches good quality at 3 to 10 steps without extra machinery. A diffusion model can match it with DDIM, but has a trap (Section 2.2) to avoid.
3. **Check the decision points.** Sample many chunks for a state where the demonstrations disagree and look at the spread. This catches weak conditioning (Section 3.1) and too few integration steps (Section 5.1) before any robot time is spent.
4. **Do not over-read small evaluations.** Report Wilson intervals (Chapter 6).

## Beyond this book

Two directions build on these chapters. Consistency models and consistency distillation compress a diffusion or flow policy to one or two steps [3]. Vision language action models, such as pi0, put a flow or diffusion action head on top of a pretrained vision language backbone [2]. The action head is the policy of this chapter.

## References

1. C. Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* RSS 2023. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)
2. K. Black et al. *pi0: A Vision-Language-Action Flow Model for General Robot Control.* 2024. [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)
3. A. Prasad et al. *Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation.* RSS 2024. [arXiv:2405.07503](https://arxiv.org/abs/2405.07503)
4. A. Tong et al. *TorchCFM*, conditional flow matching library. [GitHub](https://github.com/atong01/conditional-flow-matching)
