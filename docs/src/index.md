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

### Lasso (L1-penalized) Ordinary Least Squares Regression:
```jldoctest
julia> using DataFrames, Lasso

julia> data = DataFrame(X=[1.0,2.0,3.0], Y=[2.0,4.0,7.0])
3×2 DataFrame
│ Row │ X     │ Y     │
│     │ Int64 │ Int64 │
├─────┼───────┼───────┤
│ 1   │ 1     │ 2     │
│ 2   │ 2     │ 4     │
│ 3   │ 3     │ 7     │

julia> m = fit(LassoModel, @formula(Y ~ X), data)
StatsModels.DataFrameRegressionModel{GLM.LinearModel{GLM.LmResp{Array{Float64,1}},GLM.DensePredChol{Float64,Base.LinAlg.Cholesky{Float64,Array{Float64,2}}}},Array{Float64,2}}

Formula: Y ~ 1 + X

Coefficients:
              Estimate Std.Error  t value Pr(>|t|)
(Intercept)  -0.666667   0.62361 -1.06904   0.4788
X                  2.5  0.288675  8.66025   0.0732

julia> stderror(ols)
2-element Array{Float64,1}:
 0.62361
 0.288675

julia> predict(ols)
3-element Array{Float64,1}:
 1.83333
 4.33333
 6.83333

```
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
