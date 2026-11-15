---
title: "3.3 Training and Deployment"
parent: "3. Diffusion Policy"
nav_order: 3
---

# 3.3 Training and Deployment
{: .no_toc }

1. TOC
{:toc}

## What changes from Section 3.1

| | Section 3.1 | This section |
|:--|:--|:--|
| Denoiser | MLP | 1D temporal U-Net with FiLM (Section 3.2) |
| Observation | current position | last $$T_o = 2$$ positions |
| Action | displacement | absolute next position |
| Horizons | $$T_p = 8$$, $$T_a = 4$$ | $$T_p = 16$$, $$T_a = 8$$ |
| Normalisation | divide by max abs | min max to $$[-1, 1]$$ per dimension |
| Weights at test time | raw | exponential moving average |

These are the defaults of the Diffusion Policy paper [1], scaled down for a 2D task.

## The full script

{% include codefile.html file="ch03_train.py" %}

## Windowing with padding

For a time index $$t$$ inside an episode of length $$n$$, the sample uses observations $$t - T_o + 1, \dots, t$$ and actions $$t - T_o + 1, \dots, t - T_o + T_p$$. Indices before 0 or after $$n - 1$$ are clamped to the edge, which repeats the first observation at the start and the last action at the end.

```text
T_o = 2, T_p = 16, t = 0:
obs indices      [ 0,  0]
action indices   [ 0,  0,  1,  2, ..., 14]
                   ^ executed from here, positions T_o - 1 = 1 to T_o + T_a - 2 = 8
```

Clamping at the end is what makes the robot stop at the goal. The last action of a demonstration is the goal position, so a window that runs off the end asks the policy to keep commanding the goal.

## Normalisation

Every observation and action dimension is mapped to $$[-1, 1]$$ using the minimum and maximum over the training set. Two reasons make this more than a convenience. The DDIM clip of Section 2.2 assumes data in $$[-1, 1]$$. And the noise $$\boldsymbol\epsilon \sim \N(0, I)$$ has unit scale in every dimension, so dimensions with very different ranges, such as metres and radians, would otherwise be destroyed at very different rates by the forward process.

{: .warning }
> Save the normaliser with the checkpoint. A policy loaded with a normaliser fitted on a different dataset runs, produces plausible looking actions, and fails for no visible reason.

## EMA weights

`EMA` keeps a slowly moving copy of the weights, $$\bar\theta \leftarrow \lambda\bar\theta + (1-\lambda)\theta$$, and that copy is used for sampling. Diffusion training losses are noisy because each step sees a random noise level. Averaging the weights over the last few hundred steps gives a noticeably better sampler at no training cost [1, 2]. The reference implementation ramps $$\lambda$$ up during training. A fixed $$\lambda = 0.995$$ is enough for 3000 steps.

## The control loop

`DiffusionPolicy.act` is called once per replanning step.

1. Push the newest observation into a deque of length $$T_o$$. On reset, fill it with copies of the first observation, which matches the padding used in training.
2. Normalise and flatten the observations into the condition vector.
3. Sample a normalised chunk of shape $$(T_p, 2)$$ with 10 DDIM steps and clipping.
4. Unnormalise and return positions $$T_o - 1$$ to $$T_o + T_a - 2$$.

## Results

```text
iter     0  loss 1.2987
iter   500  loss 0.0385
iter  1000  loss 0.0138
iter  1500  loss 0.0120
iter  2000  loss 0.0132
iter  2500  loss 0.0098
UNet Diffusion Policy: {'success': 46, 'collision': 4}
```

The U-Net policy reaches 46 of 50 with 10 DDIM steps. An earlier run without a fixed evaluation seed reached 49. Training took about five minutes on a two core CPU.

Compared with 43 of 50 for the MLP policy of Section 3.1, this is **not** a significant difference at 50 trials. The toy task is simply too easy to separate them. The value of this section is the structure. Observation history, a longer action horizon, normalisation and EMA are what carry over to tasks with images and long horizons, where an MLP denoiser on raw inputs would not scale. The exercise below shows which parts matter on the toy task.

{: .exercise }
> Set `OBS_HORIZON = 1` and retrain, then set it back and use `ACTION_HORIZON = 2`. Which change costs more success? Relate your answer to the mode switching argument in the chapter introduction.

## References

1. C. Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* RSS 2023. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137). Code: [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy)
2. Y. Song, S. Ermon. *Improved Techniques for Training Score-Based Generative Models.* NeurIPS 2020. Discusses EMA for score models. [arXiv:2006.09011](https://arxiv.org/abs/2006.09011)
