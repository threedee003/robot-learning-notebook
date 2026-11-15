---
title: "2.1 DDPM from Scratch"
parent: "2. Diffusion Models"
nav_order: 1
---

# 2.1 DDPM from Scratch
{: .no_toc }

1. TOC
{:toc}

## The target distribution

We start with an unconditional problem. The data are 2D points drawn from eight Gaussians on a ring of radius 2. It is a standard toy because a model that averages, or that drops modes, is easy to spot.

## The shared core

All diffusion code in this book lives in one file. It holds the cosine schedule, a sinusoidal embedding for the step index, and a `NoiseSchedule` class with the loss and two samplers. In the code the diffusion step is called `t`, as in most libraries. In the text it is $$k$$.

{% include codefile.html file="diffusion_core.py" %}

Map each method to the equations of Chapter 2.

| Code | Equation |
|:--|:--|
| `add_noise` | $$\mathbf{x}_k = \sqrt{\bar\alpha_k}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_k}\,\boldsymbol\epsilon$$, equation (2.1) |
| `training_loss` | $$\mathcal{L}_{\text{simple}}$$ with $$k$$ uniform over $$\{0, \dots, K-1\}$$ |
| `ddpm_sample`, `mean` | $$\tilde{\boldsymbol\mu}_k$$ from the predicted noise, equation (2.2) |
| `ddpm_sample`, `var` | $$\tilde\beta_k = \frac{1-\bar\alpha_{k-1}}{1-\bar\alpha_k}\beta_k$$ |
| `ddim_sample` | Section 2.2 |

The step embedding is the same sinusoidal encoding used for positions in transformers. A scalar index becomes a vector of sines and cosines at geometrically spaced frequencies, which lets an MLP resolve both coarse and fine differences between steps.

## The denoiser and the training loop

The denoiser is a four layer MLP. It takes the noisy point, the step embedding and, optionally, a condition vector. The condition gets its own small MLP before concatenation. Chapter 3 explains why that matters.

{% include codefile.html file="ch02_ddpm_2d.py" %}

`mode_stats` counts how many samples land within 0.4 of a ring centre, and how many of the eight centres receive at least one such sample. A model that averages modes scores low on the first number. A model that drops modes scores low on the second.

## Results

Training takes about 30 seconds on a laptop CPU. The script also prints DDIM results, which Section 2.2 discusses.

```text
iter     0  loss 0.9657
iter  1000  loss 0.4485
iter  2000  loss 0.3481
iter  3000  loss 0.4055
DDPM 100 steps  on-mode 0.981  modes covered 8/8
```

The stochastic sampler with all 100 steps places 98% of samples on a mode and covers all eight. The animation shows the reverse process. The cloud starts as a single Gaussian, splits into a blurry ring as $$k$$ falls, and only resolves the eight clusters during the last fifth of the steps. This is the ladder from blurry to sharp estimates described in Chapter 2.

{% include figure.html src="/assets/media/ch02/ddpm_reverse.gif" alt="Animation of 1500 points denoised from a Gaussian into eight clusters" caption="Reverse diffusion on the eight Gaussians, as a GIF." width="60%" %}

The same animation as a video file, which is smaller and can be paused.

{% include video.html src="/assets/media/ch02/ddpm_reverse.mp4" poster="/assets/media/ch02/ddpm_final.png" caption="Reverse diffusion as an MP4 with controls." width="60%" %}

{: .exercise }
> Replace `cosine_beta_schedule` with the linear schedule of Ho et al., `torch.linspace(1e-4, 0.02, num_steps)`. Print `alpha_bar[-1]` for 100 steps. Why is $$\mathbf{x}_K$$ then not close to pure noise, and what does that do to samples that start from $$\N(0, I)$$?

## References

1. J. Ho, A. Jain, P. Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
2. A. Nichol, P. Dhariwal. *Improved Denoising Diffusion Probabilistic Models.* ICML 2021. [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)
3. A. Vaswani et al. *Attention Is All You Need.* NeurIPS 2017. Sinusoidal position encoding. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
