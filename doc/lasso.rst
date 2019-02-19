Lasso paths
=============================================

.. function:: fit(LassoPath, X, y, d=Normal(), l=canonicallink(d); ...)

    Fits a linear or generalized linear Lasso path given the design
    matrix `X` and response `y`:

    .. math:: \underset{\beta}{\operatorname{argmin}} -\frac{1}{N} \mathcal{L}(y|X,\beta) + \lambda\left[(1-\alpha)\frac{1}{2}\|\beta\|_2^2 + \alpha\|\beta\|_1\right]

    The optional argument `d` specifies the conditional distribution of
    response, while `l` specifies the link function. Lasso.jl inherits
    supported distributions and link functions from GLM.jl. The default
    is to fit an linear Lasso path, i.e., `d=Normal(), l=IdentityLink()`,
    or :math:`\mathcal{L}(y|X,\beta) = -\frac{1}{2}\|y - X\beta\|_2^2 + C`

    **Keyword arguments:**

    ================= =============================================================== ====================
      name              description                                                     default
    ================= =============================================================== ====================
     wts              Weights for each observation                                    ``ones(length(y))``
    ----------------- --------------------------------------------------------------- --------------------
     offset           Offset of each observation                                      ``zeros(length(y))``
    ----------------- --------------------------------------------------------------- --------------------
     α                Elastic Net parameter in interval [0, 1]. Controls the tradeoff ``1``
                      between L1 and L2 regularization. α = 1 fits a pure Lasso
                      model, while α = 0 would fit a pure ridge regression model.

                      **Note**: Do not set α = 0. There are methods for fitting pure
                      ridge regression models that are substantially more efficient
                      than the coordinate descent procedure used in Lasso.jl.
    ----------------- --------------------------------------------------------------- --------------------
     λ, nλ, λminratio Control the values of λ along path at which models are fit.     ``nλ = 100``

                      λ can be used to specify a specific set of λ values at which    If more observations
                      models should be fit. If λ is unspecified, Lasso.jl selects nλ  than predictors,
                      logarithmically spaced λ values from                            ``λminratio = 1e-4``.
                      :math:`\lambda_{\text{max}}`, the smallest λ value yielding a   Otherwise,
                      null model, to                                                  ``λminratio = 0.001``.
                      :math:`\lambda\text{minratio} * \lambda_{\text{max}}`.
    ----------------- --------------------------------------------------------------- --------------------
    stopearly         When set to `true`, if the                                      ``true``
                      proportion of  deviance explained exceeds 0.999 or the
                      difference between the deviance explained by successive λ
                      values falls below :math:`10^{-5}`, the path stops early.
    ----------------- --------------------------------------------------------------- --------------------
    standardize       Whether to standardize predictors to unit standard deviation    ``true``
                      before fitting.
    ----------------- --------------------------------------------------------------- --------------------
    intercept         Whether to fit an (unpenalized) model intercept.                ``true``
    ----------------- --------------------------------------------------------------- --------------------
    algorithm         Algorithm to use. The NaiveCoordinateDescent algorithm, which   NaiveCoordinateDescent
                      iteratively computes the dot product of the predictors with the if more than 5x as
                      residuals, as opposed to the CovarianceCoordinateDescent        many predictors as
                      algorithm, which uses a precomputed Gram matrix.                observations or
                      NaiveCoordinateDescent is typically faster when there are many  model is a GLM.
                      predictors that will not enter the model or when fitting        CovarianceCoordinateDescent
                      generalized linear models.                                      otherwise.
    ----------------- --------------------------------------------------------------- --------------------
    randomize         Whether to randomize the order in which coefficients are        ``true`` (if julia >= 0.4)
                      updated by coordinate descent. This can drastically speed
                      convergence if coefficients are highly correlated, but is only
                      supported under Julia 0.4.
    ----------------- --------------------------------------------------------------- --------------------
    maxncoef          The maximum number of coefficients allowed in the model. If     ``min(size(X, 2), 2*size(X, 1))``
                      exceeded, an error will be thrown.
    ----------------- --------------------------------------------------------------- --------------------
    dofit             Whether to fit the model upon construction. If `false`, the     ``true``
                      model can be fit later by calling `fit!(model)`.
    ----------------- --------------------------------------------------------------- --------------------
    cd_tol            The tolerance for coordinate descent iterations iterations in   ``1e-7``
                      the inner loop.
    ----------------- --------------------------------------------------------------- --------------------
    irls_tol          The tolerance for outer iteratively reweighted least squares    ``1e-7``
                      iterations. This is ignored unless the model is a generalized
                      linear model.
    ----------------- --------------------------------------------------------------- --------------------
    criterion         Convergence criterion. Controls how ``cd_tol`` and ``irls_tol`` ``:coef``
                      are to be interpreted. Possible values are:

                      - ``:coef``: The model is considered to have converged if the
                        the maximum absolute squared difference in coefficients
                        between successive iterations drops below the specified
                        tolerance. This is the criterion used by glmnet.
                      - ``:obj``: The model is considered to have converged if the
                        the relative change in the Lasso/Elastic Net objective
                        between successive iterations drops below the specified
                        tolerance. This is the criterion used by GLM.jl.
    ----------------- --------------------------------------------------------------- --------------------
    minStepFac        The minimum step fraction for backtracking line search.         ``0.001``
    ----------------- --------------------------------------------------------------- --------------------
    penalty_factor    Separate penalty factor :math:`\omega_j` for each coefficient   ``ones(size(X, 2))``
                      :math:`j`, i.e. instead of :math:`\lambda` penalties become
                      :math:`\lambda\omega_j`.
                      Note the penalty factors are internally rescaled to sum to
                      the number of variables (following glmnet convention).
    ----------------- --------------------------------------------------------------- --------------------
    standardizeω      Whether to scale penalty factors to sum to the number of        ``true``
                      variables (following glmnet convention).
    ================= =============================================================== ====================

    ``fit`` returns a LassoPath object describing the fit coefficients
    and values of λ along the Lasso path. The following fields are
    intended for external use:

    ================= ====================================================================================
      field              description
    ================= ====================================================================================
     λ                Vector of λ values corresponding to each fit model along the path
    ----------------- ------------------------------------------------------------------------------------
     coefs            SparseMatrixCSC of model coefficients. Columns correspond to fit models; rows
                      correspond to predictors
    ----------------- ------------------------------------------------------------------------------------
     b0               Vector of model intercepts for each fit model
    ----------------- ------------------------------------------------------------------------------------
     pct_dev          Vector of proportion of deviance explained values for each fit model
    ----------------- ------------------------------------------------------------------------------------
     nulldev          The deviance of the null model (including the intercept, if specified)
    ----------------- ------------------------------------------------------------------------------------
     nullb0           The intercept of the null model, or 0 if no intercept was fit
    ----------------- ------------------------------------------------------------------------------------
     niter            Total number of coordinate descent iterations required to fit all models
    ================= ====================================================================================

    For details of the algorithm, see Friedman, J., Hastie, T., &
    Tibshirani, R. (2010). Regularization paths for generalized linear
    models via coordinate descent. Journal of Statistical Software,
    33(1), 1.
