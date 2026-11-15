---
title: "4. Flow Matching"
nav_order: 5
has_children: true
---

# 4. Flow Matching
{: .no_toc }

<p class="rlb-meta">Prerequisites: Chapter 2, ordinary differential equations</p>

1. TOC
{:toc}

## Generation as transport

Diffusion builds a sample by removing noise over many stochastic steps. Flow matching asks a simpler question. Is there a smooth velocity field that carries noise to data along a deterministic path?

Let $$\mathbf{x}_0 \sim p_0 = \N(0, I)$$ and evolve it with an ordinary differential equation on the time interval $$\tau \in [0, 1]$$.

$$
\frac{d\mathbf{x}_\tau}{d\tau} = v_\theta(\mathbf{x}_\tau, \tau)
\tag{4.1}\label{eq:ode}
$$

The ODE moves every point, so it also moves the whole distribution. The density $$p_\tau$$ of $$\mathbf{x}_\tau$$ obeys the **continuity equation**, which says that probability mass is conserved as it flows.

$$
\frac{\partial p_\tau}{\partial \tau} + \nabla \cdot \big(p_\tau\, v_\tau\big) = 0
$$

A model defined this way is a continuous normalizing flow [1]. The goal is a field $$v_\theta$$ whose flow ends at $$p_1 = p_{\text{data}}$$. Classical training integrated the ODE inside the loss, which is slow. Flow matching removes the ODE from training entirely [2].

## The flow matching objective

Suppose we already knew a target field $$u_\tau(\mathbf{x})$$ that generates a probability path from $$p_0$$ to $$p_{\text{data}}$$. Then we could regress onto it.

$$
\mathcal{L}_{\text{FM}}(\theta) = \E_{\tau,\; \mathbf{x} \sim p_\tau}\Big[\big\lVert v_\theta(\mathbf{x}, \tau) - u_\tau(\mathbf{x})\big\rVert^2\Big]
$$

We do not know $$u_\tau$$. The key move is to build it out of simple per sample paths. Fix a data point $$\mathbf{x}_1$$ and connect it to noise with a straight line.

$$
\mathbf{x}_\tau = (1-\tau)\,\mathbf{x}_0 + \tau\,\mathbf{x}_1
\qquad\Longrightarrow\qquad
u_\tau(\mathbf{x}_\tau \mid \mathbf{x}_1) = \frac{d\mathbf{x}_\tau}{d\tau} = \mathbf{x}_1 - \mathbf{x}_0
\tag{4.2}\label{eq:linear-path}
$$

The conditional path $$p_\tau(\mathbf{x} \mid \mathbf{x}_1) = \N\big(\tau\mathbf{x}_1, (1-\tau)^2 I\big)$$ starts as pure noise and ends exactly at $$\mathbf{x}_1$$. Averaging these conditional paths over the data gives a marginal path from $$p_0$$ to $$p_{\text{data}}$$. The field that generates it is the conditional expectation of the per sample velocities.

$$
u_\tau(\mathbf{x}) = \E\big[\mathbf{x}_1 - \mathbf{x}_0 \;\big|\; \mathbf{x}_\tau = \mathbf{x}\big]
$$

## Conditional flow matching

Lipman et al. proved that regressing on the **conditional** velocity has the same gradient as regressing on the marginal one [2]. This gives a loss we can compute from samples alone.

$$
\mathcal{L}_{\text{CFM}}(\theta) = \E_{\tau \sim \mathcal{U}[0,1],\; \mathbf{x}_0 \sim \N(0,I),\; \mathbf{x}_1 \sim p_{\text{data}}}\Big[\big\lVert v_\theta\big((1-\tau)\mathbf{x}_0 + \tau\mathbf{x}_1,\; \tau\big) - (\mathbf{x}_1 - \mathbf{x}_0)\big\rVert^2\Big]
\tag{4.3}\label{eq:cfm}
$$

The proof is short. Expand both squared norms. The terms $$\lVert v_\theta \rVert^2$$ are identical. The cross terms are equal because $$\E[\langle v_\theta(\mathbf{x}_\tau), \mathbf{x}_1 - \mathbf{x}_0\rangle]$$ can be computed by first conditioning on $$\mathbf{x}_\tau$$, which replaces $$\mathbf{x}_1 - \mathbf{x}_0$$ with $$u_\tau(\mathbf{x}_\tau)$$. What remains differs by a constant that does not depend on $$\theta$$.

{: .key }
> Training is three lines. Sample noise, sample a data point, sample a time. Interpolate, and regress the network output onto the difference between data and noise. No noise schedule, no $$\bar\alpha_k$$, no posterior variances.

Sampling integrates equation \eqref{eq:ode} from $$\tau = 0$$ to $$\tau = 1$$. The simplest integrator is Euler with $$N$$ steps of size $$\Delta = 1/N$$.

$$
\mathbf{x}_{\tau + \Delta} = \mathbf{x}_\tau + \Delta\, v_\theta(\mathbf{x}_\tau, \tau)
$$

## Why this does not average modes either

The optimal network is $$v^\star(\mathbf{x}, \tau) = \E[\mathbf{x}_1 - \mathbf{x}_0 \mid \mathbf{x}_\tau = \mathbf{x}]$$, a conditional mean once more. As with the noise target in Chapter 2, the mean is taken over **directions at a point on the path**, not over final answers. Near $$\tau = 0$$ the field points towards the data mean. As $$\tau$$ grows, the point has already moved towards one mode, the conditional distribution of $$\mathbf{x}_1$$ becomes narrow, and the field points at that mode. Different starting noise therefore ends in different modes.

Marginal paths cannot cross. An ODE with a Lipschitz field has unique solutions, so two trajectories that meet would have to coincide. The per sample straight lines of equation \eqref{eq:linear-path} do cross each other, and the marginal field resolves every crossing by averaging. This is why learned trajectories are curved even though every training path is straight. Section 4.2 attacks that curvature directly.

## The connection to diffusion

Both methods use Gaussian probability paths of the form $$\mathbf{x}_\tau = \alpha_\tau \mathbf{x}_1 + \sigma_\tau \mathbf{x}_0$$. Flow matching with the linear path uses $$\alpha_\tau = \tau$$ and $$\sigma_\tau = 1 - \tau$$. The variance preserving diffusion of Chapter 2 corresponds to $$\alpha = \sqrt{\bar\alpha}$$ and $$\sigma = \sqrt{1-\bar\alpha}$$ under a change of time variable. Using Tweedie's formula $$\E[\mathbf{x}_0 \mid \mathbf{x}_\tau] = -\sigma_\tau \nabla \log p_\tau$$, the optimal velocity and the score are related linearly.

$$
v^\star(\mathbf{x}, \tau) = \frac{\dot\alpha_\tau}{\alpha_\tau}\,\mathbf{x} + \left(\frac{\dot\alpha_\tau\,\sigma_\tau}{\alpha_\tau} - \dot\sigma_\tau\right)\sigma_\tau\, \nabla_{\mathbf{x}} \log p_\tau(\mathbf{x})
$$

So a trained diffusion model already defines a velocity field, and sampling it with DDIM is an ODE solve [3]. The practical differences are in the path, the parameterisation and the time weighting, not in the class of models. The same linear path was proposed independently as rectified flow [4] and as stochastic interpolants [5].

| | Diffusion (DDPM, $$\epsilon$$ prediction) | Flow matching (linear path) |
|:--|:--|:--|
| Network predicts | added noise $$\boldsymbol\epsilon$$ | velocity $$\mathbf{x}_1 - \mathbf{x}_0$$ |
| Time | integer $$k \in \{0..K-1\}$$ | continuous $$\tau \in [0, 1]$$ |
| Schedule | $$\beta_k$$, $$\bar\alpha_k$$ tables | none |
| Sampler | stochastic (DDPM) or ODE (DDIM) | ODE (Euler, midpoint, RK4) |
| Numerical trap | division by $$\sqrt{\bar\alpha_k}$$ near pure noise | none on the linear path |

## References

1. R. T. Q. Chen, Y. Rubanova, J. Bettencourt, D. Duvenaud. *Neural Ordinary Differential Equations.* NeurIPS 2018. [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)
2. Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, M. Le. *Flow Matching for Generative Modeling.* ICLR 2023. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
3. J. Song, C. Meng, S. Ermon. *Denoising Diffusion Implicit Models.* ICLR 2021. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)
4. X. Liu, C. Gong, Q. Liu. *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.* ICLR 2023. [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
5. M. S. Albergo, E. Vanden-Eijnden. *Building Normalizing Flows with Stochastic Interpolants.* ICLR 2023. [arXiv:2209.15571](https://arxiv.org/abs/2209.15571)
6. Y. Lipman et al. *Flow Matching Guide and Code.* 2024. [arXiv:2412.06264](https://arxiv.org/abs/2412.06264)
