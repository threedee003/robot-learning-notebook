---
title: "3.2 The Temporal U-Net"
parent: "3. Diffusion Policy"
nav_order: 2
---

# 3.2 The Temporal U-Net
{: .no_toc }

1. TOC
{:toc}

## Why convolve over time

An action chunk is a short time series of shape $$T_p \times d_a$$. Neighbouring actions are strongly correlated, because a robot arm cannot teleport. A 1D convolution over the time axis builds exactly this prior into the network. Each output depends on a local window of actions, and the U-Net structure adds a coarse to fine hierarchy over the whole chunk. Janner et al. introduced this design for trajectory diffusion [1]. Diffusion Policy adds FiLM conditioning [2, 3].

We treat the action dimensions as **channels** and time as the **length** of the signal. A chunk of 16 steps for a 7 DoF arm becomes a tensor of shape `(batch, 7, 16)`.

## Building blocks

| Block | What it does | Shape change |
|:--|:--|:--|
| `Conv1dBlock` | Conv1d, GroupNorm, Mish | channels $$c_{in} \to c_{out}$$ |
| `FiLMResBlock` | Two conv blocks with FiLM after the first, plus a residual connection | channels $$c_{in} \to c_{out}$$ |
| Downsample | Conv1d with stride 2 | length $$L \to L/2$$ |
| Upsample | ConvTranspose1d with stride 2 | length $$L \to 2L$$ |

The residual block is the heart of it. Given features $$\mathbf{h}$$ and condition $$\mathbf{c}$$,

$$
\mathbf{h}' = \mathrm{Conv}_2\Big(\gamma(\mathbf{c}) \odot \mathrm{Conv}_1(\mathbf{h}) + \beta(\mathbf{c})\Big) + \mathrm{Skip}(\mathbf{h}).
$$

The condition $$\mathbf{c}$$ concatenates an embedding of the diffusion step $$k$$ with the flattened observation features. One linear layer maps $$\mathbf{c}$$ to $$2 c_{out}$$ numbers, split into $$\gamma$$ and $$\beta$$, which broadcast over time.

## The full network

{% include codefile.html file="ch03_unet.py" %}

The forward pass follows the usual U-Net pattern. On the way down it stores the output of each level, then halves the length. On the way up it concatenates the matching stored features, which is why the first residual block of each up level takes $$2 c_{out}$$ channels.

```text
input  (B, 16, 7)  -> transpose -> (B, 7, 16)
down 0 (B, 256, 16) -> stride 2 -> (B, 256, 8)
down 1 (B, 512, 8)  -> stride 2 -> (B, 512, 4)
down 2 (B, 1024, 4)               (no downsample at the last level)
mid    (B, 1024, 4)
up 0   cat (B, 2048, 4) -> (B, 512, 4) -> upsample -> (B, 512, 8)
up 1   cat (B, 1024, 8) -> (B, 256, 8) -> upsample -> (B, 256, 16)
final  (B, 7, 16) -> transpose -> (B, 16, 7)
```

Running the file prints the output shape and size.

```text
output (4, 16, 7)
params 64.8M
```

{: .warning }
> Two downsampling stages mean $$T_p$$ must be divisible by 4. With $$T_p = 10$$ the lengths become 10, 5, 3 on the way down. On the way up the length 3 is upsampled to 6, which cannot be concatenated with the stored skip of length 5, and PyTorch raises a shape error. Pad the chunk or change the number of levels.

{: .note }
> The skip from the highest resolution level is stored but never used. This mirrors the reference implementation of Diffusion Policy. It costs nothing and keeps checkpoint layouts compatible if you port weights.

## Sizing the network

The default channel widths `(256, 512, 1024)` are for image based tasks with dozens of demonstrations. For low dimensional state tasks a much smaller network trains faster with no loss in quality. Section 3.3 uses `(64, 128, 256)` with a 64 dimensional step embedding.

{: .exercise }
> Replace `nn.GroupNorm(groups, c_out)` with `nn.BatchNorm1d(c_out)` in `Conv1dBlock` and train with the EMA from Section 3.3. Compare the loss of the raw model and of the EMA model on a held out batch. Explain the gap in terms of running statistics.

## References

1. M. Janner, Y. Du, J. Tenenbaum, S. Levine. *Planning with Diffusion for Flexible Behavior Synthesis.* ICML 2022. [arXiv:2205.09991](https://arxiv.org/abs/2205.09991)
2. C. Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* RSS 2023. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)
3. E. Perez et al. *FiLM: Visual Reasoning with a General Conditioning Layer.* AAAI 2018. [arXiv:1709.07871](https://arxiv.org/abs/1709.07871)
4. O. Ronneberger, P. Fischer, T. Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI 2015. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
