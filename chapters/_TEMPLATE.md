---
title: "N.M Section Title"
parent: "N. Chapter Title"
nav_order: 1
---

# N.M Section Title
{: .no_toc }

1. TOC
{:toc}

## First heading

Plain text. Inline math $$\E[\mathbf{x}]$$. Display math below.

$$
\mathcal{L}(\theta) = \E\big[\lVert \boldsymbol\epsilon - \boldsymbol\epsilon_\theta(\mathbf{x}_k, k) \rVert^2\big]
$$

```python
import torch

x = torch.randn(4, 16, 7)
```

{: .key }
> One sentence to remember.

{% comment %}
{% include figure.html src="/assets/media/chNN/figure.png" alt="..." caption="..." %}
{% include video.html src="/assets/media/chNN/clip.mp4" caption="..." %}
{% include codefile.html file="my_script.py" %}
{% endcomment %}

## References

1.
