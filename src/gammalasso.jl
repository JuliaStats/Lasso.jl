# Implements the Taddy (2016)
# One-step estimator paths for concave regularization. arXiv
# Preprint 	arXiv:1308.5623v8. Retrieved from
# http://arxiv.org/abs/1308.5623

## GAMMA LASSO PATH

mutable struct GammaLassoPath{S<:Union{LinearModel,GeneralizedLinearModel},T} <: RegularizationPath{S,T}
    m::S
    nulldev::T                    # null deviance
    nullb0::T                     # intercept of null model, if one was fit
    λ::Vector{T}                  # shrinkage parameters
    autoλ::Bool                   # whether λ is automatically determined
    γ::Vector{T}                  # controls the concavity of the regularization path (γ=0 is Lasso) with size(X,2)
    penalty_factor::Union{Vector{T},Nothing} # regularization weights for each coeficient (multiplied by ω), defaults to 1 vector with size(X,2)
    Xnorm::Vector{T}              # original squared norms of columns of X before standardization
    pct_dev::Vector{T}            # percent deviance explained by each model
    coefs::SparseMatrixCSC{T,Int} # model coefficients
    b0::Vector{T}                 # model intercepts
    niter::Int                    # number of coordinate descent iterations

    GammaLassoPath{S,T}(m, nulldev::T, nullb0::T, λ::Vector{T}, autoλ::Bool, γ::Vector{T}, penalty_factor::Union{Vector,Nothing}, Xnorm::Vector{T}) where {S,T} =
        new(m, nulldev, nullb0, λ, autoλ, γ, penalty_factor, Xnorm)
end

copyω!(ω::Vector{T}, penalty_factor::Nothing) where T = fill!(ω, one(T))
copyω!(ω::Vector{T}, penalty_factor::Vector{T}) where T = copyto!(ω, penalty_factor)

function computeω!(ω::Vector{T}, γ::Vector{T}, penalty_factor::Union{Nothing,Vector{T}}, β::SparseCoefficients{T}) where T
    # initialize to penalty_factor
    copyω!(ω, penalty_factor)

    # set weights of non zero betas
    @inbounds @simd for icoef = 1:nnz(β)
        ipred = β.coef2predictor[icoef]
        γi = γ[ipred]
        if γi != 0.0
            ω[ipred] /= (1.0+γi*abs(β.coef[icoef]))
        end
    end

    # rescaling is done by penalty_factor, nothing about it in Taddy (2016)
    nothing
end
poststep(path::GammaLassoPath, cd::CoordinateDescent, i::Int, coefs::SparseCoefficients) = computeω!(cd.ω, path.γ, path.penalty_factor, coefs)

"""
    fit(GammaLassoPath, X, y, d=Normal(), l=canonicallink(d); ...)   
    
fits a linear or generalized linear (concave) gamma lasso path given the design
matrix `X` and response `y`.

See also [`fit(LassoPath...)`](@ref) for a full list of arguments
"""
function StatsBase.fit(::Type{GammaLassoPath},
                       X::AbstractMatrix{T}, y::V, d::UnivariateDistribution=Normal(),
                       l::Link=canonicallink(d);
                       γ::Union{Number,Vector{Number}}=0.0,
                       wts::Union{FPVector,Nothing}=ones(T, length(y)),
                       offset::AbstractVector=similar(y, 0),
                       α::Number=one(eltype(y)), nλ::Int=100,
                       λminratio::Number=ifelse(size(X, 1) < size(X, 2), 0.01, 1e-4),
                       λ::Union{Vector,Nothing}=nothing, standardize::Bool=true,
                       intercept::Bool=true,
                       algorithm::Type=defaultalgorithm(d, l, size(X, 1), size(X, 2)),
                       dofit::Bool=true,
                       irls_tol::Real=1e-7, randomize::Bool=RANDOMIZE_DEFAULT,
                       rng::Union{AbstractRNG, Nothing}=nothing,
                       maxncoef::Int=min(size(X, 2), 2*size(X, 1)),
                       penalty_factor::Union{Vector,Nothing}=nothing,
                       standardizeω::Bool=true,
                       fitargs...) where {T<:AbstractFloat,V<:FPVector}

    size(X, 1) == size(y, 1) || DimensionMismatch("number of rows in X and y must match")
    n = length(y)
    length(wts) == n || error("length(wts) = $(length(wts)) should be 0 or $n")

    X, Xnorm = standardizeX(X, standardize)

    # Gamma lasso adaptation
    # Can potentially pass a different γ for each element of X, but if scalar we copy it to all params
    p = size(X, 2)
    if isa(γ,Number)
        γ = fill(T(γ), p)
    else
        length(γ)==p || throw(DimensionMismatch("length(γ) != number of parameters ($p)"))
    end

    # Initialize penalty factors to 1 only if not supplied otherwise rescale as in glmnet
    penalty_factor = initpenaltyfactor(penalty_factor, p, standardizeω)
    ω = (isa(penalty_factor, Nothing) ? ones(T,p) : deepcopy(penalty_factor))

    # Lasso initialization
    α = convert(T, α)
    λminratio = convert(T, λminratio)
    coefitr = randomize ? RandomCoefficientIterator(rng) : (1:0)
    cd = algorithm{T,intercept,typeof(X),typeof(coefitr),typeof(ω)}(X, α, maxncoef, 1e-7, coefitr, ω)

    # GLM response initialization
    autoλ = λ == nothing
    model, nulldev, nullb0, λ = build_model(X, y, d, l, cd, λminratio, λ, wts .* T(1/sum(wts)),
                                            Vector{T}(offset), α, nλ, penalty_factor, intercept, irls_tol, dofit)

    # Fit path
    path = GammaLassoPath{typeof(model),T}(model, nulldev, nullb0, λ, autoλ, γ, penalty_factor, Xnorm)
    if dofit
        fit!(path; irls_tol=irls_tol, fitargs...)
    end
    path
end
