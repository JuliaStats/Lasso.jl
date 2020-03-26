# Lasso.jl

Lasso.jl is a pure Julia implementation of the glmnet coordinate
descent algorithm for fitting linear and generalized linear Lasso and
Elastic Net models, as described in:

[Friedman, J., Hastie, T., & Tibshirani, R. (2010)](http://www.jstatsoft.org/v33/i01/). Regularization paths
for generalized linear models via coordinate descent. Journal of
Statistical Software, 33(1), 1.

Lasso.jl also includes an implementation of the O(n) fused Lasso
implementation described in:

[Johnson, N. A. (2013)](https://doi.org/10.1080/10618600.2012.681238). A dynamic programming algorithm for the fused
lasso and L0-segmentation. Journal of Computational and Graphical
Statistics, 22(2), 246–260.

As well as an implementation of polynomial trend filtering based on:

[Ramdas, A., & Tibshirani, R. J. (2014)](http://arxiv.org/abs/1406.2082). Fast and flexible ADMM
algorithms for trend filtering. arXiv Preprint arXiv:1406.2082.

Also implements the Gamma Lasso, a concave regularization path glmnet variant based on:

[Taddy, M. (2017)](http://dx.doi.org/10.1080/10618600.2016.1211532). One-Step Estimator Paths for Concave Regularization
Journal of Computational and Graphical Statistics, 26:3, 525-536.

## Quick start

### Lasso (L1-penalized) Ordinary Least Squares Regression

```jldoctest demo1
julia> using DataFrames, Lasso

julia> data = DataFrame(X=[1,2,3], Y=[2,4,7])
3×2 DataFrames.DataFrame
│ Row │ X     │ Y     │
│     │ Int64 │ Int64 │
├─────┼───────┼───────┤
│ 1   │ 1     │ 2     │
│ 2   │ 2     │ 4     │
│ 3   │ 3     │ 7     │
```

Let's fit this to a model

``
Y = x_1 + x_2 X
``

for some scalar coefficients ``x_1`` and ``x_2``. The least-squares answer is ``x_2 = 2.5``
and ``x_1 = -2/3``,
but with lasso regularization you penalize the magnitude of `x2`. Consequently,

```jldoctest demo1
julia> m = fit(LassoModel, @formula(Y ~ X), data)
LassoModel using MinAICc(2) segment of the regularization path.

Coefficients:
────────────
    Estimate
────────────
x1  3.88915 
x2  0.222093
────────────

julia> predict(m)
3-element Array{Float64,1}:
 4.111240223622052
 4.333333333333333
 4.555426443044614

julia> predict(m, data[2:end,:])
2-element Array{Union{Missing, Float64},1}:
 4.333333333333333
 4.555426443044614
```

In the variant above, it automatically picks the size of penalty to apply to ``x_2``.

To get an entire Lasso regularization path (thus examining the consequences of a range
of penalties) with default parameters:

```julia
fit(LassoPath, X, y, dist, link)
```

where `X` is now the [design matrix](https://en.wikipedia.org/wiki/Design_matrix),
omitting the column of 1s allowing for the intercept, and `y` is the vector of
values to be fit.

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
To fit a Gamma Lasso path:

```julia
fit(GammaLassoPath, X, y, dist, link; γ=1.0)
```
It supports the same parameters as fit(LassoPath...), plus γ which controls
the concavity of the regularization path. γ=0.0 is the Lasso. Higher values
tend to result in sparser coefficient estimates.

## TODO

 - User-specified weights are untested
 - Maybe integrate LARS.jl

## See also

 - [LassoPlot.jl](https://github.com/AsafManela/LassoPlot.jl), a package for
   plotting regularization paths.
 - [GLMNet.jl](https://github.com/JuliaStats/GLMNet.jl), a wrapper for the
   glmnet Fortran code.
 - [LARS.jl](https://github.com/simonster/LARS.jl), an implementation
   of least angle regression for fitting entire linear (but not
   generalized linear) Lasso and Elastic Net coordinate paths.
