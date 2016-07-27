module Lasso
using Compat
module Util
    # Extract fields from object into function locals
    # See https://github.com/JuliaLang/julia/issues/9755
    macro extractfields(from, fields...)
        esc(Expr(:block, [:($(fields[i]) = $(from).$(fields[i])) for i = 1:length(fields)]...))
    end
    export @extractfields
end

import Base.LinAlg.BlasReal
include("FusedLasso.jl")
include("TrendFiltering.jl")

using Reexport, StatsBase, .Util
@reexport using GLM, Distributions, .FusedLassoMod, .TrendFiltering
using GLM.FPVector, GLM.wrkwt!
export RegularizationPath, LassoPath, GammaLassoPath, fit, fit!, coef, minAICc, hasintercept

## HELPERS FOR SPARSE COEFFICIENTS

immutable SparseCoefficients{T} <: AbstractVector{T}
    coef::Vector{T}              # Individual coefficient values
    coef2predictor::Vector{Int}  # Mapping from indices in coef to indices in original X
    predictor2coef::Vector{Int}  # Mapping from indices in original X to indices in coef

    SparseCoefficients(n::Int) = new(T[], Int[], zeros(Int, n))
end

function Base.A_mul_B!{T}(out::Vector, X::Matrix, coef::SparseCoefficients{T})
    fill!(out, zero(eltype(out)))
    @inbounds for icoef = 1:nnz(coef)
        ipred = coef.coef2predictor[icoef]
        c = coef.coef[icoef]
        @simd for i = 1:size(X, 1)
            out[i] += c*X[i, ipred]
        end
    end
    out
end

function Base.A_mul_B!{T}(out::Vector, X::SparseMatrixCSC, coef::SparseCoefficients{T})
    @extractfields X colptr rowval nzval
    fill!(out, zero(eltype(out)))
    @inbounds for icoef = 1:nnz(coef)
        ipred = coef.coef2predictor[icoef]
        c = coef.coef[icoef]
        @simd for i = colptr[ipred]:colptr[ipred+1]-1
            out[rowval[i]] += c*nzval[i]
        end
    end
    out
end

function Base.dot{T}(x::Vector{T}, coef::SparseCoefficients{T})
    v = 0.0
    @inbounds @simd for icoef = 1:nnz(coef)
        v += x[coef.coef2predictor[icoef]]*coef.coef[icoef]
    end
    v
end

Base.size(x::SparseCoefficients) = (length(x.predictor2coef),)
Base.nnz(x::SparseCoefficients) = length(x.coef)
Base.getindex{T}(x::SparseCoefficients{T}, ipred::Int) =
    x.predictor2coef[ipred] == 0 ? zero(T) : x.coef[x.predictor2coef[ipred]]

function Base.setindex!{T}(A::Matrix{T}, coef::SparseCoefficients, rg::UnitRange{Int}, i::Int)
    A[:, i] = zero(T)
    for icoef = 1:nnz(coef)
        A[rg[coef.coef2predictor[icoef]], i] = coef.coef[icoef]
    end
    A
end

function Base.copy!(x::SparseCoefficients, y::SparseCoefficients)
    length(x) == length(y) || throw(DimensionMismatch())
    n = length(y.coef)
    resize!(x.coef, n)
    resize!(x.coef2predictor, n)
    copy!(x.coef, y.coef)
    copy!(x.coef2predictor, y.coef2predictor)
    copy!(x.predictor2coef, y.predictor2coef)
    x
end

# Add a new coefficient to x, returning its index in x.coef
function addcoef!{T}(x::SparseCoefficients{T}, ipred::Int)
    push!(x.coef, zero(T))
    push!(x.coef2predictor, ipred)
    coefindex = nnz(x)
    x.predictor2coef[ipred] = coefindex
end

# Add newcoef to column i of coefs
function addcoefs!(coefs::SparseMatrixCSC, newcoef::SparseCoefficients, i::Int)
    n = nnz(coefs)
    nzval = coefs.nzval
    rowval = coefs.rowval
    resize!(nzval, n+nnz(newcoef))
    resize!(rowval, n+nnz(newcoef))
    @inbounds for ipred = 1:length(newcoef.predictor2coef)
        icoef = newcoef.predictor2coef[ipred]
        if icoef != 0
            cval = newcoef.coef[icoef]
            if cval != 0
                n += 1
                nzval[n] = cval
                rowval[n] = ipred
            end
        end
    end
    resize!(nzval, n)
    resize!(rowval, n)
    coefs.colptr[i+1:end] = n+1
end

## COEFFICIENT ITERATION IN SEQUENTIAL OR RANDOM ORDER

if VERSION >= v"0.4-dev+1915"
    # Julia 0.4 has a nice interface that lets us do random coefficient
    # iteration quickly.
    immutable RandomCoefficientIterator
        rng::MersenneTwister
        rg::Base.Random.RangeGeneratorInt{Int,@compat UInt}
        coeforder::Vector{Int}
    end
    const RANDOMIZE_DEFAULT = true

    RandomCoefficientIterator() =
        RandomCoefficientIterator(MersenneTwister(1337), Base.Random.RangeGenerator(1:2), Int[])
else
    immutable RandomCoefficientIterator
        RandomCoefficientIterator() = error("randomization not supported on Julia 0.3")
    end
    const RANDOMIZE_DEFAULT = false
end

typealias CoefficientIterator @compat Union{UnitRange{Int},RandomCoefficientIterator}

# Iterate over coefficients in random order
function Base.start(x::RandomCoefficientIterator)
    if !isempty(x.coeforder)
        @inbounds for i = length(x.coeforder):-1:2
            j = rand(x.rng, x.rg)
            x.coeforder[i], x.coeforder[j] = x.coeforder[j], x.coeforder[i]
        end
    end
    return 1
end
Base.next(x::RandomCoefficientIterator, i) = (x.coeforder[i], i += 1)
Base.done(x::RandomCoefficientIterator, i) = i > length(x.coeforder)

# Add an additional coefficient and return a new CoefficientIterator
function addcoef(x::RandomCoefficientIterator, icoef::Int)
    push!(x.coeforder, icoef)
    RandomCoefficientIterator(x.rng, Base.Random.RangeGenerator(1:length(x.coeforder)), x.coeforder)
end
addcoef(x::UnitRange{Int}, icoef::Int) = 1:length(x)+1

abstract RegularizationPath <: RegressionModel
## LASSO PATH

type LassoPath{S<:@compat(Union{LinearModel,GeneralizedLinearModel}),T} <: RegularizationPath
    m::S
    nulldev::T                    # null deviance
    nullb0::T                     # intercept of null model, if one was fit
    λ::Vector{T}                  # shrinkage parameters
    autoλ::Bool                   # whether λ is automatically determined
    Xnorm::Vector{T}              # original squared norms of columns of X before standardization
    pct_dev::Vector{T}            # percent deviance explained by each model
    coefs::SparseMatrixCSC{T,Int} # model coefficients
    b0::Vector{T}                 # model intercepts
    niter::Int                    # number of coordinate descent iterations

    LassoPath(m, nulldev::T, nullb0::T, λ::Vector{T}, autoλ::Bool, Xnorm::Vector{T}) =
        new(m, nulldev, nullb0, λ, autoλ, Xnorm)
end

function Base.show(io::IO, path::LassoPath)
    prefix = isa(path.m, GeneralizedLinearModel) ? string(typeof(path.m.rr.d).name.name, " ") : ""
    println(io, prefix*"Lasso Solution Path ($(size(path.coefs, 2)) solutions for $(size(path.coefs, 1)) predictors in $(path.niter) iterations):")

    coefs = path.coefs
    ncoefs = zeros(Int, size(coefs, 2))
    for i = 1:size(coefs, 2)-1
        ncoefs[i] = coefs.colptr[i+1] - coefs.colptr[i]
    end
    ncoefs[end] = nnz(coefs) - coefs.colptr[size(coefs, 2)] + 1
    Base.showarray(io, [path.λ path.pct_dev ncoefs]; header=false)
end

## MODEL CONSTRUCTION

# Controls early stopping criteria with automatic λ
const MIN_DEV_FRAC_DIFF = 1e-5
const MAX_DEV_FRAC = 0.999

# Compute automatic λ values based on X'y and λminratio
function computeλ(Xy, λminratio, α, nλ, ω::@compat(Union{Vector,Void}))
    λmax = abs(Xy[1])
    if ω != nothing
        λmax /= ω[1]
    end
    for i = 2:length(Xy)
        x = abs(Xy[i])
        if ω != nothing
            x /= ω[i]
        end
        if x > λmax
            λmax = x
        end
    end
    λmax /= α
    logλmax = log(λmax)
    λ = exp(linspace(logλmax, logλmax + log(λminratio), nλ))
end

# rescales A so that it sums to base
rescale(A,base) = A * (base / sum(A))

function StatsBase.fit{T<:AbstractFloat,V<:FPVector}(::Type{LassoPath},
                                                     X::AbstractMatrix{T}, y::V, d::UnivariateDistribution=Normal(),
                                                     l::Link=canonicallink(d);
                                                     wts::@compat(Union{FPVector,Void})=ones(T, length(y)),
                                                     offset::V=similar(y, 0),
                                                     α::Number=one(eltype(y)), nλ::Int=100,
                                                     λminratio::Number=ifelse(size(X, 1) < size(X, 2), 0.01, 1e-4),
                                                     λ::@compat(Union{Vector,Void})=nothing, standardize::Bool=true,
                                                     intercept::Bool=true,
                                                     naivealgorithm::Bool=(!isa(d, Normal) || !isa(l, IdentityLink) || size(X, 2) > 5*size(X, 1)),
                                                     dofit::Bool=true,
                                                     irls_tol::Real=1e-7, randomize::Bool=RANDOMIZE_DEFAULT,
                                                     maxncoef::Int=min(size(X, 2), 2*size(X, 1)),
                                                     penalty_factor::@compat(Union{Vector,Void})=nothing, fitargs...)
    size(X, 1) == size(y, 1) || DimensionMismatch("number of rows in X and y must match")
    n = length(y)
    length(wts) == n || error("length(wts) = $(length(wts)) should be 0 or $n")

    # Standardize predictors if requested
    if standardize
        Xnorm = vec(full(std(X, 1, corrected=false)))
        for i = 1:length(Xnorm)
            @inbounds Xnorm[i] = 1/Xnorm[i]
        end
        X = scale(X, Xnorm)
    else
        Xnorm = T[]
    end

    # Lasso initialization
    α = convert(T, α)
    λminratio = convert(T, λminratio)
    coefitr = randomize ? RandomCoefficientIterator() : (1:0)

    # penalty_factor (ω) defaults to a vector of ones
    ω = penalty_factor
    if ω != nothing
        # following glmnet rescale penalty factors to sum to the number of coefficients
        ω = rescale(ω,size(X, 2))
    end

    cd = naivealgorithm ? NaiveCoordinateDescent{T,intercept,typeof(X),typeof(coefitr)}(X, α, maxncoef, 1e-7, coefitr, ω) :
                          CovarianceCoordinateDescent{T,intercept,typeof(X),typeof(coefitr)}(X, α, maxncoef, 1e-7, coefitr, ω)

    # GLM response initialization
    autoλ = λ == nothing
    wts .*= convert(T, 1/sum(wts))
    off = convert(Vector{T}, offset)

    if isa(d, Normal) && isa(l, IdentityLink)
        # Special no-IRLS case
        mu = isempty(offset) ? y : y + off
        nullb0 = intercept ? mean(mu, weights(wts)) : zero(T)
        nulldev = 0.0
        @simd for i = 1:length(mu)
            @inbounds nulldev += abs2(mu[i] - nullb0)*wts[i]
        end

        if autoλ
            # Find max λ
            if intercept
                muscratch = Array(T, length(mu))
                @simd for i = 1:length(mu)
                    @inbounds muscratch[i] = (mu[i] - nullb0)*wts[i]
                end
            else
                muscratch = mu.*wts
            end
            Xy = X'muscratch
            λ = computeλ(Xy, λminratio, α, nλ, ω)
        else
            λ = convert(Vector{T}, λ)
        end

        # First y is just a placeholder here
        model = LinearModel(LmResp{typeof(y)}(mu, off, wts, y), cd)
    else
        # Fit to find null deviance
        # Maybe we should reuse this GlmResp object?
        nullmodel = fit(GeneralizedLinearModel, ones(T, n, ifelse(intercept, 1, 0)), y, d, l;
                        wts=wts, offset=offset, convTol=irls_tol, dofit=dofit)
        nulldev = deviance(nullmodel)

        if autoλ
            # Find max λ
            Xy = X'*broadcast!(*, nullmodel.rr.wrkresid, nullmodel.rr.wrkresid, nullmodel.rr.wrkwts)
            λ = computeλ(Xy, λminratio, α, nλ, ω)
            nullb0 = intercept ? coef(nullmodel)[1] : zero(T)
        else
            λ = convert(Vector{T}, λ)
            nullb0 = zero(T)
        end

        eta = GLM.initialeta!(d, l, similar(y), y, wts, off)
        rr = GlmResp{typeof(y),typeof(d),typeof(l)}(y, d, l, eta, similar(eta), offset, wts)
        model = GeneralizedLinearModel(rr, cd, false)
    end

    # Fit path
    path = LassoPath{typeof(model),T}(model, nulldev, nullb0, λ, autoλ, Xnorm)
    if dofit
        fit!(path; irls_tol=irls_tol, fitargs...)
    else
        path.λ = zeros(T, 0)
        path.pct_dev = zeros(T, 0)
        path.coefs = spzeros(T, p, 0)
        path.b0 = zeros(T, 0)
        path.niter = 0
    end
    path
end

StatsBase.nobs(path::RegularizationPath) = length(path.m.rr.y)
StatsBase.deviance(path::RegularizationPath) = (1 .- path.pct_dev) .* (path.nulldev * nobs(path))
dispersion_parameter(path::RegularizationPath) = typeof(path.m) <: LinearModel || GLM.dispersion_parameter(path.m.rr.d)

function StatsBase.loglikelihood(path::RegularizationPath)
    if typeof(path.m) <: LinearModel
        n = nobs(path)
        -0.5.*n.*log(deviance(path)./n)
    else
        -0.5*deviance(path)
    end
end

if Pkg.installed("StatsBase") >= v"0.8.0"
    import StatsBase.df, StatsBase.aicc
end

""" Approximates the degrees-of-freedom in each segment of the path as the number of non zero coefficients
    plus a dispersion parameter when appropriate.
    Note that for GammaLassoPath this may be a crude approximation, as gamlr does this differently.
"""
function df(path::RegularizationPath)
    nλ = length(path.λ)
    βs = coef(path)
    dof = zeros(Int,nλ)
    for s=1:nλ
        dof[s] = sum(βs[:,s].!=0)
    end

    if dispersion_parameter(path)
        # add one for dispersion_parameter
        dof+=1
    end

    dof
end

function aicc(path::RegularizationPath;k=2)
    d = df(path)
    n = nobs(path)
    ic = -2loglikelihood(path) + k*d + k*d.*(d+1)./(n-d-1)
    ic[d.+1 .> n] = realmax(eltype(ic))
    ic
end

minAICc(path::RegularizationPath;k=2)=indmin(aicc(path;k=k))

hasintercept(path::RegularizationPath) = hasintercept(path.m.pp)

#Consistent with StatsBase.coef, if the model has an intercept it is included.
function StatsBase.coef(path::RegularizationPath; select=:all)
    if length(path.λ) == 0
        X = path.m.pp.X
        p = size(X,2)
        if hasintercept(path)
            p+=1
        end
        return zeros(eltype(X),p)
    end

    if select == :all
        if hasintercept(path)
            vcat(path.b0',path.coefs)
        else
            path.coefs
        end
    elseif select == :AICc
        if hasintercept(path)
            vec(vcat(path.b0[minAICc(path)],path.coefs[:,minAICc(path)]))
        else
            path.coefs[:,minAICc(path)]
        end
    end
end

# function intercept(path::RegularizationPath; select=:all)
#     if select == :all
#         path.b0
#     elseif select == :AICc
#         path.b0[minAICc(path)]
#     end
# end

include("coordinate_descent.jl")
include("gammalasso.jl")
include("plots.jl")

end
