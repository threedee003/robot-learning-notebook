---
title: "1.1 The Multimodality Problem"
parent: "1. Imitation Learning"
nav_order: 1
---

# 1.1 The Multimodality Problem
{: .no_toc }

1. TOC
{:toc}

## A task with two correct answers

We use one toy task for the whole book. A point robot starts at the bottom of the plane and must reach a goal at the top. A round obstacle sits in the middle. Every demonstration drives straight up to a fork just below the obstacle, then a coin flip decides whether it goes around on the left or on the right.

{% include figure.html src="/assets/media/ch01/demos.png" alt="200 demonstrations that share a straight segment and then split around a circular obstacle" caption="200 demonstrations. All of them pass through the same fork point, then half go left and half go right." width="70%" %}

The observation is the 2D position $$\mathbf{o} = (x, y)$$. The action is the displacement to the next waypoint, $$\mathbf{a} = \mathbf{o}_{t+1} - \mathbf{o}_t$$. The important detail is that every demonstration visits exactly the same point on the fork. At that point the conditional action distribution has two separated peaks.

$$
p(\mathbf{a} \mid \mathbf{o}_{\text{fork}}) \approx \tfrac{1}{2}\,\delta(\mathbf{a} - \mathbf{a}_{\text{left}}) + \tfrac{1}{2}\,\delta(\mathbf{a} - \mathbf{a}_{\text{right}})
$$

The environment code is shared by every chapter.

{% include codefile.html file="toy_env.py" %}

## Behaviour cloning with MSE

The policy is a two hidden layer MLP trained with mean squared error. We normalise actions by their largest absolute value so the regression targets sit in $$[-1, 1]$$.

{% include codefile.html file="ch01_bc_mse.py" %}

Running `python ch01_bc_mse.py` prints the following on a laptop CPU.

```text
demonstrated actions at the fork (first 4):
[[ 0.058  0.031]
 [-0.074  0.037]
 [ 0.075  0.028]
 [ 0.073  0.03 ]]
epochs 100: action at fork [-0.     0.032]  rollouts {'success': 50, 'collision': 0, 'timeout': 0}
epochs 300: action at fork [0.004 0.032]  rollouts {'success': 3, 'collision': 0, 'timeout': 47}
```

## Reading the result

The demonstrators move sideways by about 0.06 to 0.075 at the fork. The trained policy moves sideways by 0.000 or 0.004. It predicts the average of the two modes, which is "straight up towards the obstacle". No demonstrator ever did that.

{% include figure.html src="/assets/media/ch01/bc_rollouts.png" alt="MSE policy rollouts for 100 and 300 training epochs" caption="Rollouts of the MSE policy. Left, 100 epochs. Right, 300 epochs. Same data, same architecture, completely different behaviour after the fork." %}

The rollouts are the more troubling part. After the averaged step the robot sits at a state that appears in no demonstration. What it does next depends only on how the network happens to extrapolate. With 100 epochs it drifts to one side and recovers in every rollout. With 300 epochs it drifts off to the right and only 3 of 50 rollouts reach the goal. Nothing in the data decides between these outcomes.

{: .warning }
> A behaviour cloning policy that works on a toy task can still be relying on accidental extrapolation. Change the seed, the number of epochs or the network width and the behaviour flips. On a real robot this shows up as a policy that "hesitates" at decision points or fails only on some evaluation days.

## The fix, in one sentence

Instead of regressing to one action, learn the whole distribution $$p(\mathbf{a} \mid \mathbf{o})$$ and **sample** from it. A sample lands in one mode or the other, never in the gap between them. Chapter 2 builds the generative model that makes this possible. Chapter 3 turns it into a policy and runs it on this same task.

{: .exercise }
> Change `side = rng.choice([-1.0, 1.0])` to always pick the right side and retrain. The MSE policy should now succeed every time, for every number of epochs. Why does removing one mode fix regression?

## References

1. C. Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* RSS 2023. Section 3 and Figure 3 discuss multimodal action distributions. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)
2. P. Florence et al. *Implicit Behavioral Cloning.* CoRL 2021. Shows the failure of explicit regression on discontinuous and multimodal targets. [arXiv:2109.00137](https://arxiv.org/abs/2109.00137)
