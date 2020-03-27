"RegularizationPath segment selector supertype"
abstract type SegSelect end

"Index of the selected RegularizationPath segment"
segselect(path::RegularizationPath, select::S; kwargs...) where S<:SegSelect =
    throw("segselect(path, ::$S) is not implemented")

"A RegularizationPath segment selector that returns all segments"
struct AllSeg <: SegSelect end

"Selects the RegularizationPath segment with the minimum AIC"
struct MinAIC <: SegSelect end

"Index of the selected RegularizationPath segment"
segselect(path::RegularizationPath, select::MinAIC; kwargs...) = minAIC(path)

"Selects the RegularizationPath segment with the minimum corrected AIC"
struct MinAICc <: SegSelect
    k::Int # k parameter used to correct AIC criterion
    MinAICc(k::Int=2) = new(k)
end
segselect(path::RegularizationPath, select::MinAICc; kwargs...) = minAICc(path; k=select.k)

"Selects the RegularizationPath segment with the minimum BIC"
struct MinBIC <: SegSelect end

segselect(path::RegularizationPath, select::MinBIC; kwargs...) = minBIC(path)

"RegularizationPath segment selector supertype"
abstract type CVSegSelect <: SegSelect end

"Selects the RegularizationPath segment with the minimum cross-validation mse"
struct MinCVmse <: CVSegSelect
    gen::CrossValGenerator

    MinCVmse(gen::CrossValGenerator) = new(gen)
    MinCVmse(path::RegularizationPath, k::Int=10) = new(Kfold(length(path.m.rr.y), k))
end

CVfun(oosdevs, ::MinCVmse) = CVmin(oosdevs)

"""
Selects the RegularizationPath segment with the largest λt with mean
OOS deviance no more than one standard error away from minimum
"""
struct MinCV1se <: CVSegSelect
    gen::CrossValGenerator

    MinCV1se(gen::CrossValGenerator) = new(gen)
    MinCV1se(path::RegularizationPath, k::Int=10) = new(Kfold(length(path.m.rr.y), k))
end

CVfun(oosdevs, ::MinCV1se) = CV1se(oosdevs)

"""
    coef(path::RegularizationPath; kwargs...)

Coefficient vector for a selected segment of a regularization path.

# Examples
```julia
coef(path; select=MinBIC())     # BIC minimizing segment
coef(path; select=AllSeg())     # Array with entire path's coefficents
```
"""
StatsBase.coef(path::RegularizationPath; select=AllSeg(), kwargs...) = coef(path, select; kwargs...)

"""
    coef(path::RegularizationPath, select::SegSelect)

Coefficient vector for a selected segment of a regularization path.

# Examples
```julia
coef(path, MinBIC())     # BIC minimizing segment
coef(path, AllSeg())     # Array with entire path's coefficents
```
"""
function StatsBase.coef(path::RegularizationPath, select::S; kwargs...) where S <: SegSelect
    if !isdefined(path,:coefs)
        X = path.m.pp.X
        p,nλ = size(path)
        return zeros(eltype(X),p)
    end

    seg = segselect(path, select; kwargs...)

    if hasintercept(path)
        vec(vcat(path.b0[seg],path.coefs[:,seg]))
    else
        path.coefs[:,seg]
    end
end

function StatsBase.coef(path::RegularizationPath, select::AllSeg; kwargs...)
    if !isdefined(path,:coefs)
        X = path.m.pp.X
        p,nλ = size(path)
        return spzeros(eltype(X),p,nλ)
    end

    if hasintercept(path)
        vcat(path.b0',path.coefs)
    else
        path.coefs
    end
end

segselect(path::RegularizationPath, select::S; kwargs...) where S<:CVSegSelect =
    cross_validate_path(path, select; kwargs...)

segselect(path::RegularizationPath,
           X::AbstractMatrix{T}, y::V,        # potentially new data
           select::S;
           kwargs...) where {T<:AbstractFloat,V<:FPVector, S<:CVSegSelect} =
    cross_validate_path(path, X, y, select; kwargs...)

"A RegularizedModel represents a selected segment from a RegularizationPath"
abstract type RegularizedModel <: RegressionModel end

"LassoModel represents a selected segment from a LassoPath"
struct LassoModel{M<:LinPredModel, S<:SegSelect} <: RegularizedModel
    lpm::M          # underlying GLM
    intercept::Bool # whether path added an intercept
    select::S       # segment selector
end

"GammaLassoModel represents a selected segment from a GammaLassoPath"
struct GammaLassoModel{M<:LinPredModel, S<:SegSelect} <: RegularizedModel
    lpm::M          # underlying GLM
    intercept::Bool # whether path added an intercept
    select::S       # segment selector
end

"""
predict(m::RegularizedModel, newX::AbstractMatrix; kwargs...)

Predicted values using a selected segment of a regularization path.

# Examples
```julia
m = fit(LassoModel, X, y; select=MinBIC())
predict(m, newX)     # predict using BIC minimizing segment
"""
function StatsBase.predict(m::RegularizedModel, newX::AbstractMatrix{T}; kwargs...) where T
    # add an interecept to newX if the model has one
    if m.intercept
        newX = [ones(T,size(newX,1),1) newX]
    end

    predict(m.lpm, newX; kwargs...)
end
StatsBase.predict(m::RegularizedModel) = predict(m.lpm)

"Returns the RegularizedPath type R used in fit(R,...)"
pathtype(::Type{LassoModel}) = LassoPath
pathtype(::Type{GammaLassoModel}) = GammaLassoPath

"""
    selectmodel(path::RegularizationPath, select::SegSelect)

Returns a LinearModel or GeneralizedLinearModel representing the selected
segment of a regularization path.

# Examples
```julia
selectmodel(path, MinBIC())            # BIC minimizing model
selectmodel(path, MinCVmse(path, 5))   # 5-fold CV mse minimizing model
```
"""
function selectmodel(path::R, select::SegSelect; kwargs...) where R<:RegularizationPath
    # extract reusable path parts
    m = path.m
    rr = deepcopy(m.rr)
    pp = m.pp
    X = pp.X 
    
    # destandardize X if needed
    if !isempty(path.Xnorm)
        X = X ./ transpose(path.Xnorm)
    end

    # add an interecept to X if the model has one
    if hasintercept(path)
        segX = [ones(eltype(X),size(X,1),1) X]
    else
        segX = X
    end

    # select coefs
    beta0 = Vector{Float64}(coef(path, select; kwargs...))

    # create new linear predictor
    pivot = true
    p = cholpred(segX, pivot)

    # rescale weights, which in GLM sum to nobs
    rr.wts .*= nobs(path)

    # same things GLM does to init just before fit!
    lp = rr.mu
    copyto!(p.beta0, beta0)
    fill!(p.delbeta, 0)
    GLM.linpred!(lp, p, 0)
    updateμ!(rr, lp)

    # create a LinearModel or GeneralizedLinearModel with the new linear predictor
    newglm(m, rr, p)
end

"""
    fit(RegularizedModel, X, y, dist, link; <kwargs>)

Returns a LinearModel or GeneralizedLinearModel representing the selected
segment of a regularization path.

# Examples
```julia
fit(LassoModel, X, y; select=MinBIC()) # BIC minimizing LinearModel
fit(LassoModel, X, y, Binomial(), Logit();
    select=MinCVmse(path, 5)) # 5-fold CV mse minimizing model
```
# Arguments
- `select::SegSelect=MinAICc()`: segment selector.
- `wts=ones(length(y))`: Weights for each observation
- `offset=zeros(length(y))`: Offset of each observation
- `λ`: can be used to specify a specific set of λ values at which models are fit.
    If λ is unspecified, Lasso.jl selects nλ logarithmically spaced λ values from
    `λmax`, the smallest λ value yielding a null model, to
    `λminratio * λmax`.
- `nλ=100` number of λ values to use
- `λminratio=1e-4` if more observations than predictors otherwise 0.001.
- `stopearly=true`: When `true`, if the proportion of deviance explained
    exceeds 0.999 or the difference between the deviance explained by successive λ
    values falls below `1e-5`, the path stops early.
- `standardize=true`: Whether to standardize predictors to unit standard deviation
    before fitting.
- `intercept=true`: Whether to fit an (unpenalized) model intercept.
- `algorithm`: Algorithm to use.
    `NaiveCoordinateDescent` iteratively computes the dot product of the
    predictors with the  residuals, as opposed to the
    `CovarianceCoordinateDescent` algorithm, which uses a precomputed Gram matrix.
    `NaiveCoordinateDescent` is typically faster when there are many
    predictors that will not enter the model or when fitting
    generalized linear models.
    By default uses `NaiveCoordinateDescent` if more than 5x as many predictors
    as observations or model is a GLM. `CovarianceCoordinateDescent` otherwise.
- `randomize=true`: Whether to randomize the order in which coefficients are
    updated by coordinate descent. This can drastically speed
    convergence if coefficients are highly correlated.
- `maxncoef=min(size(X, 2), 2*size(X, 1))`: maximum number of coefficients
    allowed in the model. If exceeded, an error will be thrown.
- `dofit=true`: Whether to fit the model upon construction. If `false`, the
    model can be fit later by calling `fit!(model)`.
- `cd_tol=1e-7`: The tolerance for coordinate descent iterations iterations in
    the inner loop.
- `irls_tol=1e-7`: The tolerance for outer iteratively reweighted least squares
    iterations. This is ignored unless the model is a generalized linear model.
- `criterion=:coef` Convergence criterion. Controls how `cd_tol` and `irls_tol`
    are to be interpreted. Possible values are:
    - `:coef`: The model is considered to have converged if the
      the maximum absolute squared difference in coefficients
      between successive iterations drops below the specified
      tolerance. This is the criterion used by glmnet.
    - `:obj`: The model is considered to have converged if the
      the relative change in the Lasso/Elastic Net objective
      between successive iterations drops below the specified
      tolerance. This is the criterion used by GLM.jl.
- `minStepFac=0.001`: The minimum step fraction for backtracking line search.
- `penalty_factor=ones(size(X, 2))`: Separate penalty factor ``\\omega_j``
    for each coefficient ``j``, i.e. instead of ``\\lambda`` penalties become
    ``\\lambda\\omega_j``.
    Note the penalty factors are internally rescaled to sum to
    the number of variables (`glmnet.R` convention).
- `standardizeω=true`: Whether to scale penalty factors to sum to the number of
    variables (glmnet.R convention).

See also [`fit(::Type{LassoPath}, ::Matrix{Float64}, ::Vector{Float64})`](@ref) for a more complete list of arguments
"""
function StatsBase.fit(::Type{R}, X::AbstractMatrix{T}, y::V,
    d::UnivariateDistribution=Normal(), l::Link=canonicallink(d);
    select::SegSelect=MinAICc(),
    intercept=true,
    kwargs...) where {R<:RegularizedModel,T<:AbstractFloat,V<:FPVector}

    # fit a regularization path
    M = pathtype(R)
    path = fit(M, X, y, d, l; intercept=intercept, kwargs...)

    R(selectmodel(path, select; kwargs...), intercept, select)
end

newglm(m::LinearModel, rr, pp) = LinearModel(rr, pp)
newglm(m::GeneralizedLinearModel, rr, pp) = GeneralizedLinearModel(rr, pp, true)

# don't add an intercept when using a @formula because we use the intercept keyword arg to add an intercept
StatsModels.drop_intercept(::Type{R}) where R<:RegularizedModel = true

StatsModels.@delegate StatsModels.TableRegressionModel.model [segselect, MinCVmse, MinCV1se]
for modeltype in (:LassoModel, :GammaLassoModel)
    @eval begin
        StatsModels.@delegate $modeltype.lpm [StatsBase.coef, StatsBase.confint,
                                     StatsBase.deviance, StatsBase.nulldeviance,
                                     StatsBase.loglikelihood, StatsBase.nullloglikelihood,
                                     StatsBase.dof, StatsBase.dof_residual, StatsBase.nobs,
                                     StatsBase.residuals, StatsBase.response
                                     ]
    end
end

# same ediom as in https://github.com/JuliaStats/GLM.jl/blob/0926a95dfc2b09179151683053de7c69e22bbe2b/src/glmfit.jl#L375
# makes converts X and y to floats
StatsBase.fit(::Type{M},
    X::AbstractMatrix,
    y::AbstractVector,
    d::UnivariateDistribution=Normal(),
    l::Link=canonicallink(d); kwargs...) where {M<:Union{RegularizationPath, RegularizedModel}} =
    fit(M, float(X), float(y), d, l; kwargs...)

function coeftable(mm::RegularizedModel)
    cc = coef(mm)
    CoefTable([cc],
                ["Estimate"],
                ["x$i" for i = 1:size(mm.lpm.pp.X, 2)])
end
    
function Base.show(io::IO, obj::RegularizedModel)
    # prefix = isa(obj.m, GeneralizedLinearModel) ? string(typeof(distfun(path)).name.name, " ") : ""
    println(io, "$(typeof(obj).name.name) using $(obj.select) segment of the regularization path.")

    println(io, "\nCoefficients:\n", coeftable(obj))
end

StatsBase.vcov(obj::RegularizedModel, args...) = error("variance-covariance matrix for a regularized model is not yet implemented")
