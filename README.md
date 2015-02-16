# Lasso

[![Build Status](https://travis-ci.org/simonster/Lasso.jl.png)](https://travis-ci.org/simonster/Lasso.jl)

Lasso.jl is a pure Julia implementation of the glmnet coordinate
descent algorithm for fitting linear and generalized linear Lasso and
Elastic Net models, as described in:

Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths
for generalized linear models via coordinate descent. Journal of
Statistical Software, 33(1), 1. http://www.jstatsoft.org/v33/i01/

## Quick start

To fit a Lasso path with default parameters:

```julia
using Lasso
fit(LassoPath, X, y, dist, link)
```

`dist` is any distribution supported by GLM.jl and `link` defaults to
the canonical link for that distribution.

More documentation is available at [ReadTheDocs](http://lassojl.readthedocs.org/en/latest/).

## TODO

 - User-specified weights are untested
 - Support unpenalized variables besides the intercept
 - Sparse matrix and custom AbstractMatrix support
 - Maybe integrate LARS.jl

## See also

 - [GLMNet.jl](https://github.com/simonster/GLMNet.jl), a wrapper for the
   glmnet Fortran code.
 - [LARS.jl](https://github.com/simonster/LARS.jl), an implementation
   of least angle regression for fitting entire linear (but not
   generalized linear) Lasso and Elastic Net coordinate paths.
