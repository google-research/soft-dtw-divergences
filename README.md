Differentiable Divergences between Time Series
==============================================

An implementation of soft-DTW divergences.

Example
-------

```python
import numpy as np
from sdtw_div.numba_ops import sdtw_div, sdtw_div_value_and_grad

# Two 3-dimensional time series of lengths 5 and 4, respectively.
X = np.random.randn(5, 3)
Y = np.random.randn(4, 3)

# Compute the divergence value. The parameter gamma controls the regularization strength. 
value = sdtw_div(X, Y, gamma=1.0)

# Compute the divergence value and the gradient w.r.t. X.
value, grad = sdtw_div_value_and_grad(X, Y, gamma=1.0)
```
Similarly, we can use `sharp_sdtw_div`, `sharp_sdtw_div_value_and_grad`,
`mean_cost_div` and `mean_cost_div_value_and_grad`.

Install
--------

Run `python setup.py install` or copy the files to your project.

Reference
----------

> Differentiable Divergences between Time Series <br/>
> Mathieu Blondel, Arthur Mensch, Jean-Philippe Vert <br/>
> [arXiv:2010.08354](https://arxiv.org/abs/2010.08354)
