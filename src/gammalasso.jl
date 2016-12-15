# using Reexport, StatsBase, ..Util, Compat
# @reexport using GLM, Lasso
# using GLM.FPVector, GLM.wrkwt!
# export GammaLassoPath

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
                                                     naivealgorithm::Bool=(!isa(d, Normal) || !isa(l, IdentityLink) || size(X, 2) > 5*size(X, 1)),
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

    # gamma lasso adaptation
    # can potentially pass a different γ for each element of X, but if scalar we copy it to all params
    p = size(X, 2)
    if isa(γ,Number)
        γ = fill(T(γ), p)
    else
        length(γ)==p || throw(DimensionMismatch("length(γ) != number of parameters ($p)"))
    end

    # initialize penalty factors to 1 (no rescaling to sum to the number of coefficients)
    ω = ones(T,p)

    # Lasso initialization
    α = convert(T, α)
    λminratio = convert(T, λminratio)
    coefitr = randomize ? RandomCoefficientIterator() : (1:0)
    cd = naivealgorithm ? NaiveCoordinateDescent{T,intercept,typeof(X),typeof(coefitr),Vector{T}}(X, α, maxncoef, 1e-7, coefitr, ω) :
                          CovarianceCoordinateDescent{T,intercept,typeof(X),typeof(coefitr),Vector{T}}(X, α, maxncoef, 1e-7, coefitr, ω)

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
            Xy = X'*broadcast!(*, nullmodel.rr.wrkresid, nullmodel.rr.wrkresid, nullmodel.rr.wrkwt)
            λ = computeλ(Xy, λminratio, α, nλ, ω)
            nullb0 = intercept ? coef(nullmodel)[1] : zero(T)
        else
            λ = convert(Vector{T}, λ)
            nullb0 = zero(T)
        end

        eta = GLM.initialeta!(d, l, similar(y), y, wts, off)
        rr = GlmResp(y, d, l, eta, similar(eta), offset, wts)
        model = GeneralizedLinearModel(rr, cd, false)
    end

    # Fit path
    path = GammaLassoPath{typeof(model),T}(model, nulldev, nullb0, λ, autoλ, γ, Xnorm)
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
