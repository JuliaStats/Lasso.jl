var documenterSearchIndex = {"docs":
[{"location":"api/#main-index-1","page":"Index","title":"Index","text":"","category":"section"},{"location":"api/#","page":"Index","title":"Index","text":"","category":"page"},{"location":"#Lasso.jl-1","page":"Home","title":"Lasso.jl","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"Lasso.jl is a pure Julia implementation of the glmnet coordinate descent algorithm for fitting linear and generalized linear Lasso and Elastic Net models, as described in:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. Journal of Statistical Software, 33(1), 1.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Lasso.jl also includes an implementation of the O(n) fused Lasso implementation described in:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Johnson, N. A. (2013). A dynamic programming algorithm for the fused lasso and L0-segmentation. Journal of Computational and Graphical Statistics, 22(2), 246–260.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"As well as an implementation of polynomial trend filtering based on:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Ramdas, A., & Tibshirani, R. J. (2014). Fast and flexible ADMM algorithms for trend filtering. arXiv Preprint arXiv:1406.2082.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Also implements the Gamma Lasso, a concave regularization path glmnet variant based on:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Taddy, M. (2017). One-Step Estimator Paths for Concave Regularization Journal of Computational and Graphical Statistics, 26:3, 525-536.","category":"page"},{"location":"#Quick-start-1","page":"Home","title":"Quick start","text":"","category":"section"},{"location":"#Lasso-(L1-penalized)-Ordinary-Least-Squares-Regression-1","page":"Home","title":"Lasso (L1-penalized) Ordinary Least Squares Regression","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"julia> using DataFrames, Lasso\n\njulia> data = DataFrame(X=[1,2,3], Y=[2,4,7])\n3×2 DataFrames.DataFrame\n Row │ X      Y\n     │ Int64  Int64\n─────┼──────────────\n   1 │     1      2\n   2 │     2      4\n   3 │     3      7","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Let's fit this to a model","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Y = x_1 + x_2 X","category":"page"},{"location":"#","page":"Home","title":"Home","text":"for some scalar coefficients x_1 and x_2. The least-squares answer is x_2 = 25 and x_1 = -23, but with lasso regularization you penalize the magnitude of x2. Consequently,","category":"page"},{"location":"#","page":"Home","title":"Home","text":"julia> m = fit(LassoModel, @formula(Y ~ X), data)\nLassoModel using MinAICc(2) segment of the regularization path.\n\nCoefficients:\n────────────\n    Estimate\n────────────\nx1  3.88915 \nx2  0.222093\n────────────\n\njulia> predict(m)\n3-element Array{Float64,1}:\n 4.111240223622052\n 4.333333333333333\n 4.555426443044614\n\njulia> predict(m, data[2:end,:])\n2-element Array{Union{Missing, Float64},1}:\n 4.333333333333333\n 4.555426443044614","category":"page"},{"location":"#","page":"Home","title":"Home","text":"In the variant above, it automatically picks the size of penalty to apply to x_2.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"To get an entire Lasso regularization path (thus examining the consequences of a range of penalties) with default parameters:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"fit(LassoPath, X, y, dist, link)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"where X is now the design matrix, omitting the column of 1s allowing for the intercept, and y is the vector of values to be fit.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"dist is any distribution supported by GLM.jl and link defaults to the canonical link for that distribution.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"To fit a fused Lasso model:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"fit(FusedLasso, y, λ)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"To fit a polynomial trend filtering model:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"fit(TrendFilter, y, order, λ)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"To fit a Gamma Lasso path:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"fit(GammaLassoPath, X, y, dist, link; γ=1.0)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"It supports the same parameters as fit(LassoPath...), plus γ which controls the concavity of the regularization path. γ=0.0 is the Lasso. Higher values tend to result in sparser coefficient estimates.","category":"page"},{"location":"#TODO-1","page":"Home","title":"TODO","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"User-specified weights are untested\nMaybe integrate LARS.jl","category":"page"},{"location":"#See-also-1","page":"Home","title":"See also","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"LassoPlot.jl, a package for plotting regularization paths.\nGLMNet.jl, a wrapper for the glmnet Fortran code.\nLARS.jl, an implementation of least angle regression for fitting entire linear (but not generalized linear) Lasso and Elastic Net coordinate paths.","category":"page"},{"location":"lasso/#Lasso-paths-1","page":"Lasso paths","title":"Lasso paths","text":"","category":"section"},{"location":"lasso/#","page":"Lasso paths","title":"Lasso paths","text":"fit(::Type{LassoPath}, ::Matrix{Float64}, ::Vector{Float64})","category":"page"},{"location":"lasso/#StatsAPI.fit-Tuple{Type{LassoPath}, Matrix{Float64}, Vector{Float64}}","page":"Lasso paths","title":"StatsAPI.fit","text":"fit(LassoPath, X, y, d=Normal(), l=canonicallink(d); ...)\n\nfits a linear or generalized linear Lasso path given the design matrix X and response y:\n\nundersetbetab_0operatornameargmin -frac1N mathcalL(yXbetab_0) + lambdaleft(1-alpha)frac12beta_2^2 + alphabeta_1right\n\nwhere 0 le alpha le 1 sets the balance between ridge (alpha = 0) and lasso (alpha = 1) regression, and N is the number of rows of X. The optional argument d specifies the conditional distribution of response, while l specifies the link function. Lasso.jl inherits supported distributions and link functions from GLM.jl. The default is to fit an linear Lasso path, i.e., d=Normal(), l=IdentityLink(), or mathcalL(yXbeta) = -frac12y - Xbeta - b_0_2^2 + C\n\nExamples\n\nfit(LassoPath, X, y)    # L1-regularized linear regression\nfit(LassoPath, X, y, Binomial(), Logit();\n    α=0.5) # Binomial logit regression with an Elastic net combination of\n           # 0.5 L1 and 0.5 L2 regularization penalties\n\nArguments\n\nwts=ones(length(y)): Weights for each observation\noffset=zeros(length(y)): Offset of each observation\nλ: can be used to specify a specific set of λ values at which models are fit.   You can pass an AbstractVector of explicit values for λ, or a function   λfunc(λmax) returning such values, where λmax will be the smallest λ   value yielding a null model.   If λ is unspecified, Lasso.jl selects nλ logarithmically spaced λ values from   λmax to λminratio * λmax.\nα=1: Value between 0 and 1 controlling the balance between ridge (alpha = 0)   and lasso (alpha = 1) regression.   α cannot be set to 0 if λ was not specified , though it may be set to 1.\nnλ=100 number of λ values to use\nλminratio=1e-4 if more observations than predictors otherwise 0.001.\nstopearly=true: When true, if the proportion of deviance explained   exceeds 0.999 or the difference between the deviance explained by successive λ   values falls below 1e-5, the path stops early.\nstandardize=true: Whether to standardize predictors to unit standard deviation   before fitting.\nintercept=true: Whether to fit an (unpenalized) model intercept b_0. If false, b_0=0.\nalgorithm: Algorithm to use.   NaiveCoordinateDescent iteratively computes the dot product of the   predictors with the  residuals, as opposed to the   CovarianceCoordinateDescent algorithm, which uses a precomputed Gram matrix.   NaiveCoordinateDescent is typically faster when there are many   predictors that will not enter the model or when fitting   generalized linear models.   By default uses NaiveCoordinateDescent if more than 5x as many predictors   as observations or model is a GLM. CovarianceCoordinateDescent otherwise.\nrandomize=true: Whether to randomize the order in which coefficients are   updated by coordinate descent. This can drastically speed   convergence if coefficients are highly correlated.\nrng=RNG_DEFAULT: Random number generator to be used for coefficient iteration.\nmaxncoef=min(size(X, 2), 2*size(X, 1)): maximum number of coefficients   allowed in the model. If exceeded, an error will be thrown.\ndofit=true: Whether to fit the model upon construction. If false, the   model can be fit later by calling fit!(model).\ncd_maxiter=100_000: The maximum number of coordinate descent iterations.\ncd_tol=1e-7: The tolerance for coordinate descent iterations iterations in   the inner loop.\nirls_maxiter=30: Maximum number of iterations in the iteratively reweighted  least squares loop. This is ignored unless the model is a generalized linear  model.\nirls_tol=1e-7: The tolerance for outer iteratively reweighted least squares   iterations. This is ignored unless the model is a generalized linear model.\ncriterion=:coef Convergence criterion. Controls how cd_tol and irls_tol   are to be interpreted. Possible values are:\n:coef: The model is considered to have converged if the the maximum absolute squared difference in coefficients between successive iterations drops below the specified tolerance. This is the criterion used by glmnet.\n:obj: The model is considered to have converged if the the relative change in the Lasso/Elastic Net objective between successive iterations drops below the specified tolerance. This is the criterion used by GLM.jl.\nminStepFac=0.001: The minimum step fraction for backtracking line search.\npenalty_factor=ones(size(X, 2)): Separate penalty factor omega_j   for each coefficient j, i.e. instead of lambda penalties become   lambdaomega_j.   Note the penalty factors are internally rescaled to sum to   the number of variables (glmnet.R convention).\nstandardizeω=true: Whether to scale penalty factors to sum to the number of   variables (glmnet.R convention).\n\n\n\n\n\n","category":"method"},{"location":"lasso/#Returned-objects-1","page":"Lasso paths","title":"Returned objects","text":"","category":"section"},{"location":"lasso/#","page":"Lasso paths","title":"Lasso paths","text":"fit returns a RegularizationPath object describing the fit coefficients and values of λ along the path. The following fields are intended for external use:","category":"page"},{"location":"lasso/#","page":"Lasso paths","title":"Lasso paths","text":"λ: Vector of λ values corresponding to each fit model along the path\ncoefs: SparseMatrixCSC of model coefficients. Columns correspond to fit models;     rows correspond to predictors\nb0: Vector of model intercepts for each fit model\npct_dev: Vector of proportion of deviance explained values for each fit model\nnulldev: The deviance of the null model (including the intercept, if specified)\nnullb0: The intercept of the null model, or 0 if no intercept was fit\nniter: Total number of coordinate descent iterations required to fit all models","category":"page"},{"location":"lasso/#","page":"Lasso paths","title":"Lasso paths","text":"For details of the algorithm, see Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. Journal of Statistical Software, 33(1), 1.","category":"page"},{"location":"lasso/#Gamma-paths-1","page":"Lasso paths","title":"Gamma paths","text":"","category":"section"},{"location":"lasso/#","page":"Lasso paths","title":"Lasso paths","text":"fit(::Type{GammaLassoPath}, ::Matrix{Float64}, ::Vector{Float64})","category":"page"},{"location":"lasso/#StatsAPI.fit-Tuple{Type{GammaLassoPath}, Matrix{Float64}, Vector{Float64}}","page":"Lasso paths","title":"StatsAPI.fit","text":"fit(GammaLassoPath, X, y, d=Normal(), l=canonicallink(d); ...)\n\nfits a linear or generalized linear (concave) gamma lasso path given the design matrix X and response y.\n\nSee also fit(LassoPath...) for a full list of arguments\n\n\n\n\n\n","category":"method"},{"location":"lasso/#Using-the-model-1","page":"Lasso paths","title":"Using the model","text":"","category":"section"},{"location":"lasso/#","page":"Lasso paths","title":"Lasso paths","text":"Lasso adheres to most of the StatsBase interface, so coef and predict should work as expected, except that a particular segment of the path would need to be selected.","category":"page"},{"location":"lasso/#","page":"Lasso paths","title":"Lasso paths","text":"coef\npredict\ndeviance\ndof\nsize","category":"page"},{"location":"lasso/#StatsAPI.coef","page":"Lasso paths","title":"StatsAPI.coef","text":"coef(path::RegularizationPath; kwargs...)\n\nCoefficient vector for a selected segment of a regularization path.\n\nExamples\n\ncoef(path; select=MinBIC())     # BIC minimizing segment\ncoef(path; select=AllSeg())     # Array with entire path's coefficents\n\n\n\n\n\ncoef(path::RegularizationPath, select::SegSelect)\n\nCoefficient vector for a selected segment of a regularization path.\n\nExamples\n\ncoef(path, MinBIC())     # BIC minimizing segment\ncoef(path, AllSeg())     # Array with entire path's coefficents\n\n\n\n\n\n","category":"function"},{"location":"lasso/#StatsAPI.predict","page":"Lasso paths","title":"StatsAPI.predict","text":"predict(path::RegularizationPath, newX::AbstractMatrix; kwargs...)\n\nPredicted values for a selected segment of a regularization path.\n\nExamples\n\npredict(path, newX; select=MinBIC())     # predict using BIC minimizing segment\n\n\n\n\n\npredicted values for data used to estimate path\n\n\n\n\n\npredict(m::RegularizedModel, newX::AbstractMatrix; kwargs...)\n\nPredicted values using a selected segment of a regularization path.\n\nExamples\n\nm = fit(LassoModel, X, y; select=MinBIC())\npredict(m, newX)     # predict using BIC minimizing segment\n\n\n\n\n\n","category":"function"},{"location":"lasso/#StatsAPI.deviance","page":"Lasso paths","title":"StatsAPI.deviance","text":"deviance at each segment of the path for the fitted model and data\n\n\n\n\n\ndeviance at each segement of the path for (potentially new) data X and y select=AllSeg() or MinAICc() like in coef()\n\n\n\n\n\ndeviance at each segment of the path for (potentially new) y and predicted values μ\n\n\n\n\n\n","category":"function"},{"location":"lasso/#StatsAPI.dof","page":"Lasso paths","title":"StatsAPI.dof","text":"dof(path::RegularizationPath)\n\nApproximates the degrees-of-freedom in each segment of the path as the number of non zero coefficients plus a dispersion parameter when appropriate. Note that for GammaLassoPath this may be a crude approximation, as gamlr does this differently.\n\n\n\n\n\n","category":"function"},{"location":"lasso/#Base.size","page":"Lasso paths","title":"Base.size","text":"size(path) returns (p,nλ) where p is the number of coefficients (including any intercept) and nλ is the number of path segments. If model was only initialized but not fit, returns (p,1).\n\n\n\n\n\n","category":"function"},{"location":"lasso/#Segment-selectors-1","page":"Lasso paths","title":"Segment selectors","text":"","category":"section"},{"location":"lasso/#","page":"Lasso paths","title":"Lasso paths","text":"SegSelect\nsegselect\nMinAIC\nMinAICc\nMinBIC\nCVSegSelect\nMinCVmse\nMinCV1se\nAllSeg","category":"page"},{"location":"lasso/#Lasso.SegSelect","page":"Lasso paths","title":"Lasso.SegSelect","text":"RegularizationPath segment selector supertype\n\n\n\n\n\n","category":"type"},{"location":"lasso/#Lasso.segselect","page":"Lasso paths","title":"Lasso.segselect","text":"Index of the selected RegularizationPath segment\n\n\n\n\n\nIndex of the selected RegularizationPath segment\n\n\n\n\n\n","category":"function"},{"location":"lasso/#Lasso.MinAIC","page":"Lasso paths","title":"Lasso.MinAIC","text":"Selects the RegularizationPath segment with the minimum AIC\n\n\n\n\n\n","category":"type"},{"location":"lasso/#Lasso.MinAICc","page":"Lasso paths","title":"Lasso.MinAICc","text":"Selects the RegularizationPath segment with the minimum corrected AIC\n\n\n\n\n\n","category":"type"},{"location":"lasso/#Lasso.MinBIC","page":"Lasso paths","title":"Lasso.MinBIC","text":"Selects the RegularizationPath segment with the minimum BIC\n\n\n\n\n\n","category":"type"},{"location":"lasso/#Lasso.CVSegSelect","page":"Lasso paths","title":"Lasso.CVSegSelect","text":"RegularizationPath segment selector supertype\n\n\n\n\n\n","category":"type"},{"location":"lasso/#Lasso.MinCVmse","page":"Lasso paths","title":"Lasso.MinCVmse","text":"Selects the RegularizationPath segment with the minimum cross-validation mse\n\n\n\n\n\n","category":"type"},{"location":"lasso/#Lasso.MinCV1se","page":"Lasso paths","title":"Lasso.MinCV1se","text":"Selects the RegularizationPath segment with the largest λt with mean OOS deviance no more than one standard error away from minimum\n\n\n\n\n\n","category":"type"},{"location":"lasso/#Lasso.AllSeg","page":"Lasso paths","title":"Lasso.AllSeg","text":"A RegularizationPath segment selector that returns all segments\n\n\n\n\n\n","category":"type"},{"location":"lasso/#Lasso-model-fitting-1","page":"Lasso paths","title":"Lasso model fitting","text":"","category":"section"},{"location":"lasso/#","page":"Lasso paths","title":"Lasso paths","text":"Often one wishes to both fit the path and select a particular segment. This can be done with fit(RegularizedModel,...), which returns a fitted RegularizedModel wrapping a GLM representation of the selected model.","category":"page"},{"location":"lasso/#","page":"Lasso paths","title":"Lasso paths","text":"For example, if we want to fit a LassoPath and select its segment that minimizes 2-fold cross-validation mean squared error, we can do it in one step as follows:","category":"page"},{"location":"lasso/#","page":"Lasso paths","title":"Lasso paths","text":"julia> using DataFrames, Lasso, MLBase, Random\n\njulia> Random.seed!(124); # because CV folds are random\n\njulia> data = DataFrame(X=[1,2,3], Y=[2,4,7])\n3×2 DataFrames.DataFrame\n Row │ X      Y\n     │ Int64  Int64\n─────┼──────────────\n   1 │     1      2\n   2 │     2      4\n   3 │     3      7\n\njulia> m = fit(LassoModel, @formula(Y ~ X), data; select=MinCVmse(Kfold(3,2)))\nLassoModel using MinCVmse(Kfold([3, 1, 2], 2, 1.5)) segment of the regularization path.\n\nCoefficients:\n────────────\n    Estimate\n────────────\nx1   4.33333\nx2   0.0    \n────────────\n\njulia> coef(m)\n2-element Array{Float64,1}:\n 4.333333333333333\n 0.0              \n","category":"page"},{"location":"lasso/#","page":"Lasso paths","title":"Lasso paths","text":"RegularizedModel\nLassoModel\nGammaLassoModel\nselectmodel\nfit(::Type{<:RegularizedModel}, ::Matrix{Float64}, ::Vector{Float64})","category":"page"},{"location":"lasso/#Lasso.RegularizedModel","page":"Lasso paths","title":"Lasso.RegularizedModel","text":"A RegularizedModel represents a selected segment from a RegularizationPath\n\n\n\n\n\n","category":"type"},{"location":"lasso/#Lasso.LassoModel","page":"Lasso paths","title":"Lasso.LassoModel","text":"LassoModel represents a selected segment from a LassoPath\n\n\n\n\n\n","category":"type"},{"location":"lasso/#Lasso.GammaLassoModel","page":"Lasso paths","title":"Lasso.GammaLassoModel","text":"GammaLassoModel represents a selected segment from a GammaLassoPath\n\n\n\n\n\n","category":"type"},{"location":"lasso/#Lasso.selectmodel","page":"Lasso paths","title":"Lasso.selectmodel","text":"selectmodel(path::RegularizationPath, select::SegSelect)\n\nReturns a LinearModel or GeneralizedLinearModel representing the selected segment of a regularization path.\n\nExamples\n\nselectmodel(path, MinBIC())            # BIC minimizing model\nselectmodel(path, MinCVmse(path, 5))   # 5-fold CV mse minimizing model\n\n\n\n\n\n","category":"function"},{"location":"lasso/#StatsAPI.fit-Tuple{Type{<:RegularizedModel}, Matrix{Float64}, Vector{Float64}}","page":"Lasso paths","title":"StatsAPI.fit","text":"fit(RegularizedModel, X, y, dist, link; <kwargs>)\n\nReturns a LinearModel or GeneralizedLinearModel representing the selected segment of a regularization path.\n\nExamples\n\nfit(LassoModel, X, y; select=MinBIC()) # BIC minimizing LinearModel\nfit(LassoModel, X, y, Binomial(), Logit();\n    select=MinCVmse(path, 5)) # 5-fold CV mse minimizing model\n\nArguments\n\nselect::SegSelect=MinAICc(): segment selector.\nwts=ones(length(y)): Weights for each observation\noffset=zeros(length(y)): Offset of each observation\nλ: can be used to specify a specific set of λ values at which models are fit.   If λ is unspecified, Lasso.jl selects nλ logarithmically spaced λ values from   λmax, the smallest λ value yielding a null model, to   λminratio * λmax.\nnλ=100 number of λ values to use\nλminratio=1e-4 if more observations than predictors otherwise 0.001.\nstopearly=true: When true, if the proportion of deviance explained   exceeds 0.999 or the difference between the deviance explained by successive λ   values falls below 1e-5, the path stops early.\nstandardize=true: Whether to standardize predictors to unit standard deviation   before fitting.\nintercept=true: Whether to fit an (unpenalized) model intercept.\nalgorithm: Algorithm to use.   NaiveCoordinateDescent iteratively computes the dot product of the   predictors with the  residuals, as opposed to the   CovarianceCoordinateDescent algorithm, which uses a precomputed Gram matrix.   NaiveCoordinateDescent is typically faster when there are many   predictors that will not enter the model or when fitting   generalized linear models.   By default uses NaiveCoordinateDescent if more than 5x as many predictors   as observations or model is a GLM. CovarianceCoordinateDescent otherwise.\nrandomize=true: Whether to randomize the order in which coefficients are   updated by coordinate descent. This can drastically speed   convergence if coefficients are highly correlated.\nmaxncoef=min(size(X, 2), 2*size(X, 1)): maximum number of coefficients   allowed in the model. If exceeded, an error will be thrown.\ndofit=true: Whether to fit the model upon construction. If false, the   model can be fit later by calling fit!(model).\ncd_tol=1e-7: The tolerance for coordinate descent iterations iterations in   the inner loop.\nirls_tol=1e-7: The tolerance for outer iteratively reweighted least squares   iterations. This is ignored unless the model is a generalized linear model.\ncriterion=:coef Convergence criterion. Controls how cd_tol and irls_tol   are to be interpreted. Possible values are:\n:coef: The model is considered to have converged if the the maximum absolute squared difference in coefficients between successive iterations drops below the specified tolerance. This is the criterion used by glmnet.\n:obj: The model is considered to have converged if the the relative change in the Lasso/Elastic Net objective between successive iterations drops below the specified tolerance. This is the criterion used by GLM.jl.\nminStepFac=0.001: The minimum step fraction for backtracking line search.\npenalty_factor=ones(size(X, 2)): Separate penalty factor omega_j   for each coefficient j, i.e. instead of lambda penalties become   lambdaomega_j.   Note the penalty factors are internally rescaled to sum to   the number of variables (glmnet.R convention).\nstandardizeω=true: Whether to scale penalty factors to sum to the number of   variables (glmnet.R convention).\n\nSee also fit(::Type{LassoPath}, ::Matrix{Float64}, ::Vector{Float64}) for a more complete list of arguments\n\n\n\n\n\n","category":"method"},{"location":"smoothing/#Fused-Lasso-and-trend-filtering-1","page":"Fused Lasso and trend filtering","title":"Fused Lasso and trend filtering","text":"","category":"section"},{"location":"smoothing/#","page":"Fused Lasso and trend filtering","title":"Fused Lasso and trend filtering","text":"function fit(FusedLasso, y, λ)","category":"page"},{"location":"smoothing/#","page":"Fused Lasso and trend filtering","title":"Fused Lasso and trend filtering","text":"Fits the fused Lasso model:","category":"page"},{"location":"smoothing/#","page":"Fused Lasso and trend filtering","title":"Fused Lasso and trend filtering","text":"undersetbetaoperatornameargmin frac12 sum_k=1^N(y_k - beta_k)^2 + lambda sum_k=2^N beta_k - beta_k-1","category":"page"},{"location":"smoothing/#","page":"Fused Lasso and trend filtering","title":"Fused Lasso and trend filtering","text":"The model coefficients can be obtained by calling coef on the returned model object.","category":"page"},{"location":"smoothing/#","page":"Fused Lasso and trend filtering","title":"Fused Lasso and trend filtering","text":"For details of the algorithm, see Johnson, N. A. (2013). A dynamic programming algorithm for the fused lasso and L0-segmentation. Journal of Computational and Graphical Statistics, 22(2), 246–260. doi:10.1080/10618600.2012.681238","category":"page"},{"location":"smoothing/#","page":"Fused Lasso and trend filtering","title":"Fused Lasso and trend filtering","text":"function fit(TrendFilter, y, order, λ)","category":"page"},{"location":"smoothing/#","page":"Fused Lasso and trend filtering","title":"Fused Lasso and trend filtering","text":"Fits the trend filter model:","category":"page"},{"location":"smoothing/#","page":"Fused Lasso and trend filtering","title":"Fused Lasso and trend filtering","text":"undersetbetaoperatornameargmin frac12 sum_k=1^N(y_k - beta_k)^2 + lambda D^(k+1)beta_k_1","category":"page"},{"location":"smoothing/#","page":"Fused Lasso and trend filtering","title":"Fused Lasso and trend filtering","text":"Where D^(k+1) is the discrete difference operator of order k+1. The model coefficients can be obtained by calling coef on the returned model object.","category":"page"},{"location":"smoothing/#","page":"Fused Lasso and trend filtering","title":"Fused Lasso and trend filtering","text":"For details of the algorithm, see Ramdas, A., & Tibshirani, R. J. (2014). Fast and flexible ADMM algorithms for trend filtering. arXiv Preprint arXiv:1406.2082. Retrieved from http://arxiv.org/abs/1406.2082","category":"page"}]
}
