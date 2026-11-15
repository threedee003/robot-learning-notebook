---
title: Home
layout: home
nav_order: 1
permalink: /
description: "A hands-on book on diffusion and flow matching policies for robot learning."
---

# Robot Learning Notebook
{: .fs-9 }

Generative policies for robots, built from first principles. Each chapter derives the method, then implements it in PyTorch on a problem small enough to train on a laptop CPU.
{: .fs-6 .fw-300 }

[Start reading]({{ '/chapters/01-imitation-learning/' | relative_url }}){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[How to add a chapter]({{ '/appendix/authoring-guide/' | relative_url }}){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## Chapters

<div class="rlb-chapter-grid">
  <a href="{{ '/chapters/01-imitation-learning/' | relative_url }}">
    <span class="rlb-chapter-num">Chapter 1</span>
    <span class="rlb-chapter-title">Imitation Learning</span>
    <span class="rlb-chapter-blurb">Behaviour cloning, compounding error, and why regression to the mean breaks multimodal demonstrations.</span>
  </a>
  <a href="{{ '/chapters/02-diffusion-models/' | relative_url }}">
    <span class="rlb-chapter-num">Chapter 2</span>
    <span class="rlb-chapter-title">Diffusion Models</span>
    <span class="rlb-chapter-blurb">The forward and reverse processes, the noise prediction loss, DDPM and DDIM sampling.</span>
  </a>
  <a href="{{ '/chapters/03-diffusion-policy/' | relative_url }}">
    <span class="rlb-chapter-num">Chapter 3</span>
    <span class="rlb-chapter-title">Diffusion Policy</span>
    <span class="rlb-chapter-blurb">Action chunks, receding horizon control, the FiLM conditioned temporal U-Net, and a full training loop.</span>
  </a>
  <a href="{{ '/chapters/04-flow-matching/' | relative_url }}">
    <span class="rlb-chapter-num">Chapter 4</span>
    <span class="rlb-chapter-title">Flow Matching</span>
    <span class="rlb-chapter-blurb">Continuous normalizing flows, the conditional flow matching objective, OT couplings and rectified flow.</span>
  </a>
  <a href="{{ '/chapters/05-flow-policies/' | relative_url }}">
    <span class="rlb-chapter-num">Chapter 5</span>
    <span class="rlb-chapter-title">Flow Matching Policies</span>
    <span class="rlb-chapter-blurb">Velocity fields over action chunks, time sampling, few step integration, and the design of pi0.</span>
  </a>
  <a href="{{ '/chapters/06-deployment/' | relative_url }}">
    <span class="rlb-chapter-num">Chapter 6</span>
    <span class="rlb-chapter-title">From Simulation to Robot</span>
    <span class="rlb-chapter-blurb">Latency, temporal ensembling, action representations, and how to evaluate a policy honestly.</span>
  </a>
</div>

## How to read this book

Read the chapters in order the first time. Chapters 2 and 4 are self-contained introductions to the generative models. Chapters 3 and 5 turn those models into robot policies. If you already know diffusion models, you can skip straight to Chapter 3.

Every code listing on this site is a real file in `_includes/code/`. The book page and the runnable script are the same file, so they cannot drift apart. To run them, clone the repository and use the commands below.

```bash
cd _includes/code
python -m venv .venv && source .venv/bin/activate
pip install torch numpy scipy matplotlib
python ch01_bc_mse.py
```

## Conventions

| Symbol | Meaning |
|:--|:--|
| $$\mathbf{o}_t$$ | observation at control step $$t$$ |
| $$\mathbf{a}_t$$ | action at control step $$t$$ |
| $$\mathbf{A}_t$$ | action chunk $$(\mathbf{a}_t, \dots, \mathbf{a}_{t+T_p-1})$$ |
| $$k$$ | diffusion step, integer in $$\{0, \dots, K-1\}$$ |
| $$\tau$$ | flow matching time in $$[0, 1]$$, with $$\tau = 0$$ as noise and $$\tau = 1$$ as data |
| $$T_o, T_p, T_a$$ | observation, prediction and action horizons |

Press <kbd>Ctrl</kbd>+<kbd>K</kbd> to search. Use the moon button at the top right to switch to dark mode.
