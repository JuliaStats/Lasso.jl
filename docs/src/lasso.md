# Lasso paths

```@docs
fit
```

## Returned objects
``fit`` returns a `RegularizationPath` object describing the fit coefficients
and values of λ along the path. The following fields are
intended for external use:

- `λ`: Vector of λ values corresponding to each fit model along the path
- `coefs`: SparseMatrixCSC of model coefficients. Columns correspond to fit models;
      rows correspond to predictors
- `b0`: Vector of model intercepts for each fit model
- `pct_dev`: Vector of proportion of deviance explained values for each fit model
- `nulldev`: The deviance of the null model (including the intercept, if specified)
- `nullb0`: The intercept of the null model, or 0 if no intercept was fit
- `niter`: Total number of coordinate descent iterations required to fit all models

For details of the algorithm, see Friedman, J., Hastie, T., &
Tibshirani, R. (2010). Regularization paths for generalized linear
models via coordinate descent. Journal of Statistical Software,
33(1), 1.

## Using the model
Lasso adhears to most of the `StatsBase` interface, so `coef` and `predict`
should work as expected, except that a particular segment of the path
would need to be selected.

```@docs
coef
predict
deviance
dof
size
```

## Segment selectors
```@docs
SegSelect
segselect
MinAIC
MinAICc
MinBIC
CVSegSelect
MinCVmse
MinCV1se
AllSeg
```

## Lasso model fitting
Often one wishes to both fit the path and select a particular segment.
This can be done with `fit(RegularizedModel,...)`, which returns a fitted
`RegularizedModel` wrapping a `GLM` representation of the selected model.

For example, if we want to fit a `LassoPath` and select its segment
that minimizes 2-fold cross-validation mean squared error, we can do it in one
step as follows:

```jldoctest
julia> using DataFrames, Lasso, MLBase, Random

julia> Random.seed!(124); # because CV folds are random

julia> data = DataFrame(X=[1,2,3], Y=[2,4,7])
3×2 DataFrames.DataFrame
│ Row │ X     │ Y     │
│     │ Int64 │ Int64 │
├─────┼───────┼───────┤
│ 1   │ 1     │ 2     │
│ 2   │ 2     │ 4     │
│ 3   │ 3     │ 7     │

julia> m = fit(LassoModel, @formula(Y ~ X), data; select=MinCVmse(Kfold(3,2)))
StatsModels.TableRegressionModel{LassoModel{LinearModel{GLM.LmResp{Array{Float64,1}},GLM.DensePredQR{Float64}}},Array{Float64,2}}

Y ~ X

Coefficients:
──────────────────────────────────────────────────────────────────
    Estimate  Std. Error   t value  Pr(>|t|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────
x1   4.33333      5.4365  0.797081    0.5716   -64.744     73.4106
x2   0.0          2.0548  0.0         1.0000   -26.1088    26.1088
──────────────────────────────────────────────────────────────────

julia> coef(m)
2-element Array{Float64,1}:
 4.333333333333335
 0.0              

```

```@docs
RegularizedModel
LassoModel
GammaLassoModel
selectmodel
```
