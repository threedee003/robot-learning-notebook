---
title: "2. Diffusion Models"
nav_order: 3
has_children: true
---

# 2. Diffusion Models
{: .no_toc }

<p class="rlb-meta">Prerequisites: Gaussian identities, Chapter 1</p>

1. TOC
{:toc}

## The idea

We want to sample from a data distribution $$p_{\text{data}}(\mathbf{x})$$ that we only know through samples. In our case $$\mathbf{x}$$ will be an action chunk. A diffusion model does this in two halves.

1. A fixed **forward process** slowly adds Gaussian noise to data until nothing is left but noise.
2. A learned **reverse process** removes the noise one small step at a time.

Each reverse step only has to undo a tiny amount of corruption, so each step is close to a Gaussian and easy to learn. Chaining many such steps gives a very expressive model. This chapter uses $$k$$ for the diffusion step to keep it separate from the control step $$t$$ of the robot.

## The forward process

Pick a variance schedule $$\beta_1, \dots, \beta_K \in (0, 1)$$ and define a Markov chain that shrinks the signal and adds noise.

$$
q(\mathbf{x}_k \mid \mathbf{x}_{k-1}) = \N\!\left(\mathbf{x}_k;\; \sqrt{1-\beta_k}\,\mathbf{x}_{k-1},\; \beta_k I\right)
$$

Write $$\alpha_k = 1 - \beta_k$$ and $$\bar\alpha_k = \prod_{s=1}^{k} \alpha_s$$. Because a linear map of a Gaussian plus independent Gaussian noise is again Gaussian, we can collapse $$k$$ steps into one.

$$
\begin{aligned}
\mathbf{x}_k &= \sqrt{\alpha_k}\,\mathbf{x}_{k-1} + \sqrt{1-\alpha_k}\,\boldsymbol\epsilon_k \\
&= \sqrt{\alpha_k \alpha_{k-1}}\,\mathbf{x}_{k-2} + \sqrt{\alpha_k(1-\alpha_{k-1})}\,\boldsymbol\epsilon_{k-1} + \sqrt{1-\alpha_k}\,\boldsymbol\epsilon_k
\end{aligned}
$$

The two noise terms are independent, so their variances add to $$\alpha_k(1-\alpha_{k-1}) + 1 - \alpha_k = 1 - \alpha_k\alpha_{k-1}$$. Repeating the argument down to $$\mathbf{x}_0$$ gives the closed form

$$
q(\mathbf{x}_k \mid \mathbf{x}_0) = \N\!\left(\sqrt{\bar\alpha_k}\,\mathbf{x}_0,\; (1-\bar\alpha_k) I\right),
$$

or equivalently, as a recipe for sampling,

$$
\mathbf{x}_k = \sqrt{\bar\alpha_k}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_k}\,\boldsymbol\epsilon,\qquad \boldsymbol\epsilon \sim \N(0, I).
\tag{2.1}\label{eq:forward}
$$

{: .key }
> Equation \eqref{eq:forward} is what makes training cheap. We never simulate the chain. We pick a random $$k$$, draw one $$\boldsymbol\epsilon$$, and jump straight to $$\mathbf{x}_k$$.

The schedule is chosen so that $$\bar\alpha_K \approx 0$$. Then $$\mathbf{x}_K$$ is essentially pure noise and does not depend on the data. Ho et al. used a linear schedule over $$K = 1000$$ steps [1]. Nichol and Dhariwal proposed the cosine schedule $$\bar\alpha_k = f(k)/f(0)$$ with $$f(k) = \cos^2\!\big(\tfrac{k/K + s}{1+s}\cdot\tfrac{\pi}{2}\big)$$, which destroys information more evenly across steps [2]. Diffusion Policy uses the cosine schedule with $$K = 100$$ [3], and so do we.

## The reverse process

Sampling needs $$q(\mathbf{x}_{k-1} \mid \mathbf{x}_k)$$, which depends on the unknown data distribution. However, if we also condition on $$\mathbf{x}_0$$ the posterior is Gaussian and has a closed form (apply Bayes' rule to two Gaussians).

$$
q(\mathbf{x}_{k-1} \mid \mathbf{x}_k, \mathbf{x}_0) = \N\!\left(\tilde{\boldsymbol\mu}_k,\; \tilde\beta_k I\right),
\qquad
\tilde\beta_k = \frac{1-\bar\alpha_{k-1}}{1-\bar\alpha_k}\,\beta_k
$$

$$
\tilde{\boldsymbol\mu}_k = \frac{\sqrt{\bar\alpha_{k-1}}\,\beta_k}{1-\bar\alpha_k}\,\mathbf{x}_0 + \frac{\sqrt{\alpha_k}\,(1-\bar\alpha_{k-1})}{1-\bar\alpha_k}\,\mathbf{x}_k
$$

Now substitute $$\mathbf{x}_0 = (\mathbf{x}_k - \sqrt{1-\bar\alpha_k}\,\boldsymbol\epsilon)/\sqrt{\bar\alpha_k}$$ from equation \eqref{eq:forward}. After simplification the mean depends on the data only through the noise that was added.

$$
\tilde{\boldsymbol\mu}_k = \frac{1}{\sqrt{\alpha_k}}\left(\mathbf{x}_k - \frac{\beta_k}{\sqrt{1-\bar\alpha_k}}\,\boldsymbol\epsilon\right)
\tag{2.2}\label{eq:posterior-mean}
$$

So we train a network $$\boldsymbol\epsilon_\theta(\mathbf{x}_k, k)$$ to predict that noise and plug it into equation \eqref{eq:posterior-mean}. One reverse step is

$$
\mathbf{x}_{k-1} = \frac{1}{\sqrt{\alpha_k}}\left(\mathbf{x}_k - \frac{\beta_k}{\sqrt{1-\bar\alpha_k}}\,\boldsymbol\epsilon_\theta(\mathbf{x}_k, k)\right) + \sqrt{\tilde\beta_k}\,\mathbf{z},\qquad \mathbf{z} \sim \N(0, I).
\tag{2.3}\label{eq:ddpm-step}
$$

## The training objective

The variational bound on $$-\log p_\theta(\mathbf{x}_0)$$ decomposes into one KL term per step. Each KL compares two Gaussians with the same variance, so it reduces to a squared distance between means, which in turn is a weighted squared error on $$\boldsymbol\epsilon$$. Ho et al. found that dropping the weights works better in practice [1]. The result is the loss used almost everywhere.

$$
\mathcal{L}_{\text{simple}}(\theta) = \E_{\mathbf{x}_0,\, k,\, \boldsymbol\epsilon}\Big[\big\lVert \boldsymbol\epsilon - \boldsymbol\epsilon_\theta\big(\sqrt{\bar\alpha_k}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_k}\,\boldsymbol\epsilon,\; k\big)\big\rVert^2\Big]
$$

## Why this regression does not average modes

This loss is again a mean squared error, which seems to contradict Chapter 1. The difference lies in what is being regressed. The optimal network is the conditional mean of the noise.

$$
\boldsymbol\epsilon^\star(\mathbf{x}_k, k) = \E[\boldsymbol\epsilon \mid \mathbf{x}_k] = -\sqrt{1-\bar\alpha_k}\;\nabla_{\mathbf{x}_k} \log p_k(\mathbf{x}_k)
$$

The second equality is Tweedie's formula, and it is the link to denoising score matching [4, 5]. Here $$p_k$$ is the distribution of noisy samples at step $$k$$. So the network learns the **score**, the direction in which the noisy density increases. Averaging is exactly right for this target. A score field of a bimodal density points towards the nearer peak, so following it from random noise carries each sample into one mode. In Chapter 1 we averaged **actions**. Here we average **noise**, and the average of the noise defines the score.

{: .intuition }
> At large $$k$$ the implied clean estimate $$\hat{\mathbf{x}}_0$$ is blurry and sits between modes, just like the MSE policy. At small $$k$$ the noisy point is already close to one mode and the estimate is sharp. Sampling walks down this ladder, so the early blurry guesses get corrected step by step.

## What comes next

- Section 2.1 implements the schedule, the loss and the sampler of equation \eqref{eq:ddpm-step} in about 100 lines.
- Section 2.2 derives DDIM, which samples in 10 steps instead of 100, and shows a numerical trap that it hides.

## References

1. J. Ho, A. Jain, P. Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
2. A. Nichol, P. Dhariwal. *Improved Denoising Diffusion Probabilistic Models.* ICML 2021. [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)
3. C. Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* RSS 2023. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)
4. P. Vincent. *A Connection Between Score Matching and Denoising Autoencoders.* Neural Computation 23(7), 2011.
5. Y. Song et al. *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR 2021. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
6. B. Efron. *Tweedie's Formula and Selection Bias.* Journal of the American Statistical Association 106(496), 2011.
