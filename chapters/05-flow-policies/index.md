---
title: "5. Flow Matching Policies"
nav_order: 6
has_children: true
---

# 5. Flow Matching Policies
{: .no_toc }

<p class="rlb-meta">Prerequisites: Chapters 3 and 4</p>

1. TOC
{:toc}

## The policy

A flow matching policy keeps everything from Diffusion Policy except the generative objective. The network now predicts a velocity over action chunks, conditioned on observations.

$$
\mathcal{L}(\theta) = \E_{\tau,\; \boldsymbol\epsilon \sim \N(0, I),\; (\mathbf{O}_t, \mathbf{A}_t) \sim \mathcal{D}}\Big[\big\lVert v_\theta\big(\mathbf{A}^\tau_t,\, \tau,\, \mathbf{O}_t\big) - (\mathbf{A}_t - \boldsymbol\epsilon)\big\rVert^2\Big],
\qquad
\mathbf{A}^\tau_t = (1-\tau)\,\boldsymbol\epsilon + \tau\,\mathbf{A}_t
$$

At test time we draw $$\mathbf{A}^0_t \sim \N(0, I)$$ and integrate $$d\mathbf{A}^\tau_t / d\tau = v_\theta(\mathbf{A}^\tau_t, \tau, \mathbf{O}_t)$$ to $$\tau = 1$$. The observation horizon, prediction horizon, action horizon, normaliser, EMA and network all carry over from Chapter 3 unchanged. In code the change is about twenty lines.

## Why few integration steps are enough

Consider the extreme case where the observation fully determines the action chunk, so $$p(\mathbf{A} \mid \mathbf{O}) = \delta(\mathbf{A} - \boldsymbol\mu(\mathbf{O}))$$. Then $$\mathbf{x}_1 = \boldsymbol\mu$$ for every training pair and the optimal field is

$$
v^\star(\mathbf{x}, \tau) = \E[\boldsymbol\mu - \mathbf{x}_0 \mid \mathbf{x}_\tau = \mathbf{x}] = \frac{\boldsymbol\mu - \mathbf{x}}{1 - \tau}.
$$

Every trajectory is a straight line to $$\boldsymbol\mu$$ at constant speed, and a **single** Euler step from $$\tau = 0$$ lands exactly on it. The more peaked the conditional distribution, the closer the learned field is to this case. Robot observations usually pin down the action quite tightly. Only a few decision points, like our fork, are genuinely multimodal. There the field must bend to send nearby noise samples to different modes, and that is where more steps help.

{: .key }
> Curvature in the flow comes from ambiguity in the data. Strong conditioning makes paths straight, and straight paths are cheap to integrate.

## Time sampling

The CFM loss of Chapter 4 samples $$\tau$$ uniformly. Not every time is equally hard. Near $$\tau = 1$$ the input is almost clean and the network only has to point at a target it can nearly see. Near $$\tau = 0$$ the input is almost pure noise, and the network must infer the whole chunk from the observation alone. That is the part of the problem the observation actually solves.

The pi0 model therefore samples $$\tau$$ from a shifted Beta distribution that puts more weight on noisy times [1]. In our convention this is

$$
u \sim \mathrm{Beta}(1.5,\, 1),\qquad \tau = s\,(1 - u),\qquad s = 0.999.
$$

The cap $$s < 1$$ removes times very close to clean data, which the authors note are not needed when integration uses 10 steps.

{: .warning }
> Papers and codebases do not agree on the direction of time. pi0's public code uses $$t = 1$$ for noise and integrates backwards, with target $$\boldsymbol\epsilon - \mathbf{A}$$. This book uses $$\tau = 0$$ for noise and target $$\mathbf{A} - \boldsymbol\epsilon$$. Mixing the two conventions silently produces a policy that walks away from the data. Check the sign of the target first when porting code.

## The design of pi0

pi0 [1] is the best known flow matching policy. It is a vision language action model, but its action head is exactly the policy above.

| Component | Choice in pi0 |
|:--|:--|
| Backbone | PaliGemma vision language model, about 3B parameters |
| Action expert | A separate set of transformer weights, about 300M parameters, for state and action tokens |
| Conditioning | Action tokens attend to image, language and proprioception tokens |
| Chunk length | 50 actions |
| Integration | 10 Euler steps |
| Control rate | up to 50 Hz on dexterous tasks |
| Time sampling | shifted Beta, as above |

The action expert design matters for latency. The image and language prefix is encoded once per control step and cached. Only the small action expert runs 10 times inside the ODE loop. This is the same trick as encoding $$\mathbf{O}_t$$ once in Diffusion Policy.

## Sections in this chapter

- 5.1 implements the flow policy on the fork task and measures success against the number of Euler steps.
- 5.2 compares diffusion and flow policies on cost, quality and engineering effort.

## References

1. K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. *pi0: A Vision-Language-Action Flow Model for General Robot Control.* 2024. [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)
2. Y. Lipman et al. *Flow Matching for Generative Modeling.* ICLR 2023. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
3. Physical Intelligence. *openpi*, open source code and weights for pi0. [GitHub](https://github.com/Physical-Intelligence/openpi)
