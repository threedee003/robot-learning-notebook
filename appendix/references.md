---
title: "References"
parent: "Appendix"
nav_order: 2
math: false
---

# References
{: .no_toc }

Every work cited in the book, grouped by topic. Each chapter also lists its own references at the end.

1. TOC
{:toc}

## Imitation learning

- S. Ross, J. A. Bagnell. *Efficient Reductions for Imitation Learning.* AISTATS 2010. [PMLR](https://proceedings.mlr.press/v9/ross10a.html)
- S. Ross, G. Gordon, J. A. Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.* AISTATS 2011. [arXiv:1011.0686](https://arxiv.org/abs/1011.0686)
- C. M. Bishop. *Mixture Density Networks.* Aston University technical report NCRG/94/004, 1994.
- P. Florence, C. Lynch, A. Zeng, O. Ramirez, A. Wahid, L. Downs, A. Wong, J. Lee, I. Mordatch, J. Tompson. *Implicit Behavioral Cloning.* CoRL 2021. [arXiv:2109.00137](https://arxiv.org/abs/2109.00137)
- N. M. Shafiullah, Z. J. Cui, A. Altanzaya, L. Pinto. *Behavior Transformers: Cloning k Modes with One Stone.* NeurIPS 2022. [arXiv:2206.11251](https://arxiv.org/abs/2206.11251)
- T. Z. Zhao, V. Kumar, S. Levine, C. Finn. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT).* RSS 2023. [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)

## Diffusion models

- J. Ho, A. Jain, P. Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- J. Song, C. Meng, S. Ermon. *Denoising Diffusion Implicit Models.* ICLR 2021. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)
- A. Nichol, P. Dhariwal. *Improved Denoising Diffusion Probabilistic Models.* ICML 2021. [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)
- Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, B. Poole. *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR 2021. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
- T. Karras, M. Aittala, T. Aila, S. Laine. *Elucidating the Design Space of Diffusion-Based Generative Models.* NeurIPS 2022. [arXiv:2206.00364](https://arxiv.org/abs/2206.00364)
- P. Vincent. *A Connection Between Score Matching and Denoising Autoencoders.* Neural Computation 23(7), 2011.
- B. Efron. *Tweedie's Formula and Selection Bias.* Journal of the American Statistical Association 106(496), 2011.

## Diffusion for control

- M. Janner, Y. Du, J. Tenenbaum, S. Levine. *Planning with Diffusion for Flexible Behavior Synthesis.* ICML 2022. [arXiv:2205.09991](https://arxiv.org/abs/2205.09991)
- C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, S. Song. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* RSS 2023. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137). Code: [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy)
- Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, H. Xu. *3D Diffusion Policy.* RSS 2024. [arXiv:2403.03954](https://arxiv.org/abs/2403.03954)
- A. Prasad, K. Lin, J. Wu, L. Zhou, J. Bohg. *Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation.* RSS 2024. [arXiv:2405.07503](https://arxiv.org/abs/2405.07503)

## Flow matching

- R. T. Q. Chen, Y. Rubanova, J. Bettencourt, D. Duvenaud. *Neural Ordinary Differential Equations.* NeurIPS 2018. [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)
- Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, M. Le. *Flow Matching for Generative Modeling.* ICLR 2023. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- X. Liu, C. Gong, Q. Liu. *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.* ICLR 2023. [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
- M. S. Albergo, E. Vanden-Eijnden. *Building Normalizing Flows with Stochastic Interpolants.* ICLR 2023. [arXiv:2209.15571](https://arxiv.org/abs/2209.15571)
- A. Tong, K. Fatras, N. Malkin, G. Huguet, Y. Zhang, J. Rector-Brooks, G. Wolf, Y. Bengio. *Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport.* TMLR 2024. [arXiv:2302.00482](https://arxiv.org/abs/2302.00482)
- A.-A. Pooladian, H. Ben-Hamu, C. Domingo-Enrich, B. Amos, Y. Lipman, R. T. Q. Chen. *Multisample Flow Matching: Straightening Flows with Minibatch Couplings.* ICML 2023. [arXiv:2304.14772](https://arxiv.org/abs/2304.14772)
- Y. Lipman, M. Havasi, P. Holderrieth, N. Shaul, M. Le, B. Karrer, R. T. Q. Chen, D. Lopez-Paz, H. Ben-Hamu, I. Gat. *Flow Matching Guide and Code.* 2024. [arXiv:2412.06264](https://arxiv.org/abs/2412.06264)

## Flow matching for control

- K. Black et al. *pi0: A Vision-Language-Action Flow Model for General Robot Control.* 2024. [arXiv:2410.24164](https://arxiv.org/abs/2410.24164). Code: [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)

## Architecture

- O. Ronneberger, P. Fischer, T. Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI 2015. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
- E. Perez, F. Strub, H. de Vries, V. Dumoulin, A. Courville. *FiLM: Visual Reasoning with a General Conditioning Layer.* AAAI 2018. [arXiv:1709.07871](https://arxiv.org/abs/1709.07871)
- Y. Wu, K. He. *Group Normalization.* ECCV 2018. [arXiv:1803.08494](https://arxiv.org/abs/1803.08494)
- Y. Zhou, C. Barnes, J. Lu, J. Yang, H. Li. *On the Continuity of Rotation Representations in Neural Networks.* CVPR 2019. [arXiv:1812.07035](https://arxiv.org/abs/1812.07035)

## Evaluation

- E. B. Wilson. *Probable Inference, the Law of Succession, and Statistical Inference.* Journal of the American Statistical Association 22(158), 1927.
