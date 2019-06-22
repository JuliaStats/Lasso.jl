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
This can be done with `fit(RegularizedModel,...)`, which returns a fitten GLM
representing the selected model.

```@docs
LassoModel
GammaLassoModel
selectmodel
```
