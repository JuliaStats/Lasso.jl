# Implements the Taddy (2016)
# One-step estimator paths for concave regularization. arXiv
# Preprint 	arXiv:1308.5623v8. Retrieved from
# http://arxiv.org/abs/1308.5623

## GAMMA LASSO PATH

type GammaLassoPath{S<:Union{LinearModel,GeneralizedLinearModel},T} <: RegularizationPath{S,T}
    m::S
    nulldev::T                    # null deviance
    nullb0::T                     # intercept of null model, if one was fit
    λ::Vector{T}                  # shrinkage parameters
    autoλ::Bool                   # whether λ is automatically determined
    γ::Vector{T}                  # controls the concavity of the regularization path (γ=0 is Lasso) with size(X,2)
    Xnorm::Vector{T}              # original squared norms of columns of X before standardization
    pct_dev::Vector{T}            # percent deviance explained by each model
    coefs::SparseMatrixCSC{T,Int} # model coefficients
    b0::Vector{T}                 # model intercepts
    niter::Int                    # number of coordinate descent iterations

    GammaLassoPath(m, nulldev::T, nullb0::T, λ::Vector{T}, autoλ::Bool, γ::Vector{T}, Xnorm::Vector{T}) =
        new(m, nulldev, nullb0, λ, autoλ, γ, Xnorm)
end

"Compute coefficient specific weights vector ω_j^t based on previous iteration coefficients β"
function computeω!{T}(ω::Vector{T}, γ::Vector{T}, β::SparseCoefficients{T})
    # initialize to a vector of ones
    fill!(ω,one(T))

    # set weights of non zero betas
    @inbounds @simd for icoef = 1:nnz(β)
        ipred = β.coef2predictor[icoef]
        γi = γ[ipred]
        if γi != 0.0
            ω[ipred] = 1.0/(1.0+γi*abs(β.coef[icoef]))
        end
    end

    # rescale(ω,p) # not sure if rescaling is the right thing to do here, nothing about it in Taddy (2016)
    nothing
end
poststep(path::GammaLassoPath, cd::CoordinateDescent, i::Int, coefs::SparseCoefficients) = computeω!(cd.ω, path.γ, coefs)

function StatsBase.fit{T<:AbstractFloat,V<:FPVector}(::Type{GammaLassoPath},
                                                     X::AbstractMatrix{T}, y::V, d::UnivariateDistribution=Normal(),
                                                     l::Link=canonicallink(d);
                                                     γ::Union{Number,Vector{Number}}=0.0,
                                                     wts::Union{FPVector,Void}=ones(T, length(y)),
                                                     offset::AbstractVector=similar(y, 0),
                                                     α::Number=one(eltype(y)), nλ::Int=100,
                                                     λminratio::Number=ifelse(size(X, 1) < size(X, 2), 0.01, 1e-4),
                                                     λ::Union{Vector,Void}=nothing, standardize::Bool=true,
                                                     intercept::Bool=true,
                                                     algorithm::Type=defaultalgorithm(d, l, size(X, 1), size(X, 2)),
                                                     dofit::Bool=true,
                                                     irls_tol::Real=1e-7, randomize::Bool=RANDOMIZE_DEFAULT,
                                                     maxncoef::Int=min(size(X, 2), 2*size(X, 1)), fitargs...)

    size(X, 1) == size(y, 1) || DimensionMismatch("number of rows in X and y must match")
    n = length(y)
    length(wts) == n || error("length(wts) = $(length(wts)) should be 0 or $n")

    # Standardize predictors if requested
    if standardize
        Xnorm = vec(full(std(X, 1, corrected=false)))
        for i = 1:length(Xnorm)
            @inbounds Xnorm[i] = 1/Xnorm[i]
        end
        X = X .* Xnorm.'
    else
        Xnorm = T[]
    end

    # Gamma lasso adaptation
    # Can potentially pass a different γ for each element of X, but if scalar we copy it to all params
    p = size(X, 2)
    if isa(γ,Number)
        γ = fill(T(γ), p)
    else
        length(γ)==p || throw(DimensionMismatch("length(γ) != number of parameters ($p)"))
    end

    # Initialize penalty factors to 1 (no rescaling to sum to the number of coefficients)
    ω = ones(T,p)

    # Lasso initialization
    α = convert(T, α)
    λminratio = convert(T, λminratio)
    coefitr = randomize ? RandomCoefficientIterator() : (1:0)
    cd = algorithm{T,intercept,typeof(X),typeof(coefitr),Vector{T}}(X, α, maxncoef, 1e-7, coefitr, ω)

    # GLM response initialization
    autoλ = λ == nothing
    model, nulldev, nullb0, λ = build_model(X, y, d, l, cd, λminratio, λ, wts .* T(1/sum(wts)),
                                            Vector{T}(offset), α, nλ, ω, intercept, irls_tol)

    # Fit path
    path = GammaLassoPath{typeof(model),T}(model, nulldev, nullb0, λ, autoλ, γ, Xnorm)
    if dofit
        fit!(path; irls_tol=irls_tol, fitargs...)
    end
    path
end
