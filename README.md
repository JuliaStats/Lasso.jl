# Lasso

[![Build Status](https://travis-ci.org/simonster/Lasso.jl.svg?branch=master)](https://travis-ci.org/simonster/Lasso.jl)
[![Coverage Status](https://coveralls.io/repos/simonster/Lasso.jl/badge.svg?branch=master)](https://coveralls.io/r/simonster/Lasso.jl?branch=master)

Lasso.jl is a pure Julia implementation of the glmnet coordinate
descent algorithm for fitting linear and generalized linear Lasso and
Elastic Net models, as described in:

Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths
for generalized linear models via coordinate descent. Journal of
Statistical Software, 33(1), 1. http://www.jstatsoft.org/v33/i01/

Lasso.jl also includes an implementation of the O(n) fused Lasso
implementation described in:

Johnson, N. A. (2013). A dynamic programming algorithm for the fused
lasso and L0-segmentation. Journal of Computational and Graphical
Statistics, 22(2), 246–260. doi:10.1080/10618600.2012.681238

As well as an implementation of polynomial trend filtering based on:

Ramdas, A., & Tibshirani, R. J. (2014). Fast and flexible ADMM
algorithms for trend filtering. arXiv Preprint arXiv:1406.2082.
Retrieved from http://arxiv.org/abs/1406.2082

## Quick start

To fit a Lasso path with default parameters:

```julia
fit(LassoPath, X, y, dist, link)
```

`dist` is any distribution supported by GLM.jl and `link` defaults to
the canonical link for that distribution.

To fit a fused Lasso model:

```julia
fit(FusedLasso, y, λ)
```

To fit a polynomial trend filtering model:

```julia
fit(TrendFilter, y, order, λ)
```

More documentation is available at [ReadTheDocs](http://lassojl.readthedocs.org/en/latest/).

## TODO

 - User-specified weights are untested
 - Support unpenalized variables besides the intercept
 - Maybe integrate LARS.jl

## See also

 - [GLMNet.jl](https://github.com/simonster/GLMNet.jl), a wrapper for the
   glmnet Fortran code.
 - [LARS.jl](https://github.com/simonster/LARS.jl), an implementation
   of least angle regression for fitting entire linear (but not
   generalized linear) Lasso and Elastic Net coordinate paths.
