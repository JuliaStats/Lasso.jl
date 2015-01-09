module Lasso
import Base.LinAlg.BlasReal

using Reexport, StatsBase
@reexport using GLM, Distributions
using GLM.FPVector, GLM.wrkwt!
export LassoPath, fit

## HELPERS FOR SPARSE COEFFICIENTS

immutable SparseCoefficients{T} <: AbstractVector{T}
    coef::Vector{T}
    coef2predictor::Vector{Int}
    predictor2coef::Vector{Int}

    SparseCoefficients(n::Int) = new(T[], Int[], zeros(Int, n))
end

function A_mul_B!{T}(out::Vector, X::Matrix, coef::SparseCoefficients{T})
    fill!(out, zero(eltype(out)))
    @inbounds for icoef = 1:nnz(coef)
        ipred = coef.coef2predictor[icoef]
        @simd for i = 1:size(X, 1)
            out[i] += coef.coef[icoef]*X[i, ipred]
        end
    end
    out
end

function dot{T}(x::Vector{T}, coef::SparseCoefficients{T})
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

function Base.setindex!{T}(A::Matrix{T}, coef::SparseCoefficients, rg::Range1{Int}, i::Int)
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

function addcoef!{T}(x::SparseCoefficients{T}, ipred::Int)
    push!(x.coef, zero(T))
    push!(x.coef2predictor, ipred)
    coefindex = length(x.coef2predictor)
    x.predictor2coef[ipred] = coefindex
end

## UTILITIES

# To make sure that results are reproducible on the same dataset
function shuffle!(rng::AbstractRNG, a::AbstractVector)
    @inbounds for i = length(a):-1:2
        j = rand(rng, 1:i)
        a[i], a[j] = a[j], a[i]
    end
    return a
end

## COORDINATE DESCENT ROUTINES
S(z, γ) = ifelse(γ >= abs(z), zero(z), ifelse(z > 0, z - γ, z + γ))

function P{T}(α::T, β::SparseCoefficients{T})
    x = zero(T)
    @inbounds @simd for i = 1:nnz(β)
        x += (1 - α)/2*abs2(β.coef[i]) + α*abs(β.coef[i])
    end
    x
end

abstract CoordinateDescent{T} <: LinPred
type NaiveCoordinateDescent{T} <: CoordinateDescent{T}
    X::Matrix{T}                  # original design matrix
    Xdw::Matrix{T}                # X, decentered and weighted
    μy::T                         # mean of y at current weights
    μX::Vector{T}                 # mean of X at current weights
    Xssq::Vector{T}               # weighted sum of squares of each column of X
    residuals::Vector{T}          # y - Xβ (unscaled with centered X)
    weights::Vector{T}            # weights for each observation
    weightsuminv::T               # 1/sum(weights)
    coeforder::Vector{Int}        # order of coefficients for partial passes
    rng::MersenneTwister          # RNG for shuffling coefficients
    dev::T                        # last deviance
    intercept::Bool               # whether an intercept should be fitted
    α::T                          # elastic net parameter
    maxiter::Int                  # maximum number of iterations
    tol::T                        # tolerance as ratio of deviance to null deviance

    NaiveCoordinateDescent{T}(X::Matrix{T}, intercept::Bool, α::T, maxiter::Int, tol::T)  =
        new(X, Array(T, size(X)), zero(T), zeros(T, size(X, 2)), Array(T, size(X, 2)),
            Array(T, size(X, 1)), Array(T, size(X, 1)), convert(T, NaN), Int[], MersenneTwister(1337),
            convert(T, NaN), intercept, α, maxiter, tol)
end

# Updates CoordinateDescent object with (possibly) new y vector and
# weights
function update!{T}(cd::NaiveCoordinateDescent{T}, coef::SparseCoefficients{T}, y::Vector{T}, wt::Vector{T})
    residuals = cd.residuals
    X = cd.X
    Xssq = cd.Xssq
    Xdw = cd.Xdw
    weights = copy!(cd.weights, wt)
    # weights = scale!(cd.weights, wt, 1/length(y))
    weightsuminv = cd.weightsuminv = inv(sum(weights))

    A_mul_B!(residuals, X, coef)
    @simd for i = 1:length(y)
        @inbounds residuals[i] = y[i] - residuals[i]
    end

    if cd.intercept
        μX = cd.μX
        @inbounds for j = 1:size(X, 2)
            # Update μX
            μ = zero(T)
            @simd for i = 1:size(X, 1)
                μ += X[i, j]*weights[i]
            end
            μ *= weightsuminv
            μX[j] = μ

            # Update Xssq
            ws = zero(T)
            @simd for i = 1:size(X, 1)
                xmu = X[i, j] - μ
                xmuw = Xdw[i, j] = xmu*weights[i]
                ws += xmu*xmuw
            end
            Xssq[j] = ws
        end

        # Compute μy and μres
        μy = zero(T)
        μres = zero(T)
        @simd for i = 1:length(y)
            @inbounds μy += y[i]*weights[i]
            @inbounds μres += residuals[i]*weights[i]
        end
        μy *= weightsuminv
        μres *= weightsuminv
        cd.μy = μy

        # Center residuals
        @simd for i = 1:length(residuals)
            @inbounds residuals[i] -= μres
        end
    else
        @inbounds for j = 1:size(X, 2)
            ws = zero(T)
            @simd for i = 1:size(X, 1)
                x = Xdw[i, j] = X[i, j]*weights[i]
                ws += X[i, j]*x
            end
            Xssq[j] = ws
        end

        nulldev = zero(T)
        @simd for i = 1:length(y)
            nulldev += abs2(y[i])*weights[i]
        end
    end
    cd
end

# Performs the cycle of all predictors
function cycle!{T}(coef::SparseCoefficients{T}, cd::NaiveCoordinateDescent{T}, λ::T, α::T, all::Bool)
    X = cd.X
    Xdw = cd.Xdw
    residuals = cd.residuals
    weights = cd.weights
    Xssq = cd.Xssq
    μX = cd.μX
    coeforder = cd.coeforder

    ptr = pointer(Xdw)
    maxdelta = zero(T)
    if all
        @inbounds for ipred = 1:size(X, 2)
            # Use all variables for first and last iterations

            # Update coefficient
            icoef = coef.predictor2coef[ipred]
            oldcoef = icoef == 0 ? zero(T) : coef.coef[icoef]
            v = Xssq[ipred]*oldcoef + BLAS.dot(size(X, 1), ptr+(ipred-1)*size(X, 1)*sizeof(T), 1, residuals, 1)
            newcoef = S(v, λ*α)/(Xssq[ipred] + λ*(1 - α))
            # println("s $ipred => $v, den = $((Xssq[ipred] + λ*(1 - α)))")
            # println("$ipred => $newcoef")

            # Update residual
            if oldcoef != newcoef
                if icoef == 0
                    icoef = addcoef!(coef, ipred)
                    push!(coeforder, icoef)
                end
                coef.coef[icoef] = newcoef
                cμX = μX[ipred]
                @simd for i = 1:length(residuals)
                    residuals[i] += (oldcoef - newcoef)*(X[i, ipred] - cμX)
                end
                maxdelta = max(maxdelta, abs2(oldcoef - newcoef)*Xssq[ipred])
            end
        end
    else
        shuffle!(cd.rng, coeforder)
        @inbounds for icoef = coeforder
            oldcoef = coef.coef[icoef]
            oldcoef == 0 && continue
            ipred = coef.coef2predictor[icoef]

            # Update coefficient
            v = Xssq[ipred]*oldcoef + BLAS.dot(size(X, 1), ptr+(ipred-1)*size(X, 1)*sizeof(T), 1, residuals, 1)
            newcoef = coef.coef[icoef] = S(v, λ*α)/(Xssq[ipred] + λ*(1 - α))
            # println("s $j => $v, den = $((Xssq[j] + λ*(1 - α)))")
            # println("$j => $(coef[j])")

            # Update residual
            cμX = μX[ipred]
            @simd for i = 1:length(residuals)
                residuals[i] += (oldcoef - newcoef)*(X[i, ipred] - cμX)
            end
            maxdelta = max(maxdelta, abs2(oldcoef - newcoef)*Xssq[ipred])
        end
    end
    maxdelta*cd.weightsuminv
end

# Sum of squared residuals. Residuals are always up to date
function ssr{T}(coef::SparseCoefficients{T}, cd::NaiveCoordinateDescent{T})
    residuals = cd.residuals
    weights = cd.weights
    s = zero(T)
    @simd for i = 1:length(residuals)
        @inbounds s += abs2(residuals[i])*weights[i]
    end
    s
end

type CovarianceCoordinateDescent{T} <: CoordinateDescent{T}
    X::Matrix{T}                  # original design matrix
    μy::T                         # mean of y at current weights
    μX::Vector{T}                 # mean of X at current weights
    yty::T                        # y'y (scaled by weights)
    Xty::Vector{T}                # X'y (scaled by weights)
    Xssq::Vector{T}               # weighted sum of squares of each column of X
    XtX::Matrix{T}                # X'X (scaled by weights, in order of coefficients)
    weights::Vector{T}            # weights for each observation
    weightsuminv::T               # inverse of sum of weights
    coeforder::Vector{Int}        # order of coefficients for partial passes
    rng::MersenneTwister          # RNG for shuffling coefficients
    dev::T                        # last deviance
    intercept::Bool               # whether an intercept should be fitted
    α::T                          # elastic net parameter
    maxiter::Int                  # maximum number of iterations
    tol::T                        # tolerance as ratio of deviance to null deviance

    function CovarianceCoordinateDescent{T}(X::Matrix{T}, intercept::Bool, α::T, maxiter::Int, tol::T)
        new(X, zero(T), zeros(T, size(X, 2)), convert(T, NaN), Array(T, size(X, 2)),
            Array(T, size(X, 2)), Array(T, min(5*size(X, 1), size(X, 2)), size(X, 2)), Array(T, size(X, 1)),
            convert(T, NaN), Int[], MersenneTwister(1337), convert(T, NaN), intercept, α, maxiter, tol)
    end
end

# Updates CoordinateDescent object with (possibly) new y vector and
# weights
function update!{T}(cd::CovarianceCoordinateDescent{T}, coef::SparseCoefficients{T}, y::Vector{T}, wt::Vector{T})
    X = cd.X
    Xty = cd.Xty
    μX = cd.μX
    Xssq = cd.Xssq
    XtX = cd.XtX
    weights = copy!(cd.weights, wt)
    weightsuminv = cd.weightsuminv = inv(sum(weights))
    if cd.intercept
        # Compute μy
        μy = zero(T)
        @simd for i = 1:length(y)
            @inbounds μy += y[i]*weights[i]
        end
        μy *= weightsuminv
        cd.μy = μy

        # Compute y'y
        yty = zero(T)
        @simd for i = 1:length(y)
            @inbounds yty += abs2(y[i] - μy)*weights[i]
        end

        for j = 1:size(X, 2)
            μ = zero(T)
            # Compute weighted mean
            @inbounds @simd for i = 1:size(X, 1)
                μ += X[i, j]*weights[i]
            end
            μ *= weightsuminv
            μX[j] = μ

            # Compute Xssq and X'y
            v = zero(T)
            ws = zero(T)
            @simd for i = 1:size(X, 1)
                ws += abs2(X[i, j] - μ)*weights[i]
                @inbounds v += (y[i] - μy)*(X[i, j] - μ)*weights[i]
            end
            Xssq[j] = ws
            Xty[j] = v
        end
    else
        # Compute y'y
        yty = zero(T)
        @simd for i = 1:length(y)
            @inbounds yty += abs2(y[i])*weights[i]
        end

        for j = 1:size(X, 2)
            # Compute Xssq and X'y
            v = zero(T)
            ws = zero(T)
            @simd for i = 1:size(X, 1)
                ws += abs2(X[i, j])*weights[i]
                @inbounds v += y[i]*X[i, j]*weights[i]
            end
            Xssq[j] = ws
            Xty[j] = v
        end
    end

    for icoef = 1:nnz(coef)
        ipred = coef.coef2predictor[icoef]
        for jpred = 1:size(X, 2)
            s = 0.0
            @simd for i = 1:size(X, 1)
                s += (X[i, ipred] - μX[ipred])*(X[i, jpred] - μX[jpred])*weights[i]
            end
            XtX[icoef, jpred] = s
        end
    end

    cd.yty = yty

    cd
end

# Performs the cycle of all predictors
function cycle!{T}(coef::SparseCoefficients{T}, cd::CovarianceCoordinateDescent{T}, λ::T, α::T, all::Bool)
    Xty = cd.Xty
    XtX = cd.XtX
    Xssq = cd.Xssq
    coeforder = cd.coeforder

    maxdelta = 0.0
    if all
        @inbounds for ipred = 1:length(Xty)
            # Use all variables for first and last iterations
            s = Xty[ipred]
            @simd for jcoef = 1:nnz(coef)
                s -= XtX[jcoef, ipred]*coef.coef[jcoef]
            end

            icoef = coef.predictor2coef[ipred]
            if icoef != 0
                oldcoef = coef.coef[icoef]
                s += XtX[icoef, ipred]*oldcoef
            else
                oldcoef = zero(T)
            end

            newcoef = S(s, λ*α)/(Xssq[ipred] + λ*(1 - α))
            if oldcoef != newcoef
                if icoef == 0
                    icoef = addcoef!(coef, ipred)
                    push!(coeforder, icoef)

                    # Compute cross-product with predictors
                    weights = cd.weights
                    X = cd.X
                    μX = cd.μX
                    for jpred = 1:size(X, 2)
                        s = 0.0
                        @simd for i = 1:size(X, 1)
                            s += (X[i, ipred] - μX[ipred])*(X[i, jpred] - μX[jpred])*weights[i]
                        end
                        XtX[icoef, jpred] = s
                    end
                end
                maxdelta = max(maxdelta, abs2(oldcoef - newcoef)*Xssq[ipred])
                coef.coef[icoef] = newcoef
            end
            # println("s $j => $s, den = $((XtX[j, j] + λ*(1 - α)))")
            # println("$j => $(coef[j])")
        end
    else
        shuffle!(cd.rng, coeforder)
        @inbounds for icoef = coeforder
            ipred = coef.coef2predictor[icoef]
            oldcoef = coef.coef[icoef]
            oldcoef == 0 && continue
            s = Xty[ipred] + XtX[icoef, ipred]*oldcoef
            @simd for jcoef = 1:nnz(coef)
                s -= XtX[jcoef, ipred]*coef.coef[jcoef]
            end
            newcoef = coef.coef[icoef] = S(s, λ*α)/(Xssq[ipred] + λ*(1 - α))
            maxdelta = max(maxdelta, abs2(oldcoef - newcoef)*Xssq[ipred])
        end
    end
    maxdelta
    # println()
end

# y'y - 2β'X'y + β'X'Xβ
function ssr{T}(coef::SparseCoefficients{T}, cd::CovarianceCoordinateDescent{T})
    # v = cd.yty - 2*dot(coef, cd.Xty) + dot(coef, BLAS.symv!('U', one(T), cd.XtX, coef, zero(T), cd.scratch))
    XtX = cd.XtX
    Xty = cd.Xty
    v = cd.yty
    @inbounds for icoef = 1:nnz(coef)
        ipred = coef.coef2predictor[icoef]
        s = -2*Xty[ipred]
        @simd for jcoef = 1:nnz(coef)
            s += XtX[jcoef, ipred]*coef.coef[jcoef]
        end
        v += coef.coef[icoef]*s
    end
    v
end

function fit!{T}(coef::SparseCoefficients{T}, cd::CoordinateDescent{T}, λ, criterion)
    α = cd.α
    maxiter = cd.maxiter
    tol = cd.tol
    n = size(cd.X, 1)

    obj = convert(T, Inf)
    objold = convert(T, Inf)
    dev = convert(T, Inf)
    prev_converged = false
    converged = true

    for iter = 1:maxiter
        maxdelta = cycle!(coef, cd, λ, α, converged)

        # Test for convergence
        prev_converged = converged
        if criterion == :obj
            objold = obj
            dev = ssr(coef, cd)
            obj = dev/2 + λ*P(α, coef)
            converged = objold - obj < tol*obj
            # @show obj (objold - obj)/obj
        elseif criterion == :coef
            converged = maxdelta < tol
        end

        # Require two converging steps to return. After the first, we
        # will iterate through all variables
        prev_converged && converged && break
    end

    (!prev_converged || !converged) && error("coordinate descent failed to converge in $maxiter iterations at λ = $λ")

    cd.dev = criterion == :coef ? ssr(coef, cd) : dev
    coef
end

intercept{T}(coef::SparseCoefficients{T}, cd::CoordinateDescent{T}) = cd.intercept ? cd.μy .- dot(cd.μX, coef) : zero(T)

function linpred!{T}(mu::Vector{T}, X::Matrix{T}, coef::SparseCoefficients{T}, b0::T)
    A_mul_B!(mu, X, coef)
    if b0 != 0
        @simd for i = 1:length(mu)
            @inbounds mu[i] += b0
        end
    end
    mu
end

## LASSO PATH

type LassoPath{S<:Union(LinearModel, GeneralizedLinearModel),T} <: RegressionModel
    m::S
    nulldev::T              # null deviance
    nullb0::T               # intercept of null model, if one was fit
    λ::Vector{T}            # shrinkage parameters
    autoλ::Bool             # whether λ is automatically determined
    Xnorm::Vector{T}        # original norms of columns of X before standardization
    pct_dev::Vector{T}      # percent deviance explained by each model
    coefs::Matrix{T}        # model coefficients
    b0::Vector{T}           # model intercepts

    LassoPath(m, nulldev::T, nullb0::T, λ::Vector{T}, autoλ::Bool, Xnorm::Vector{T}) =
        new(m, nulldev, nullb0, λ, autoλ, Xnorm)
end

function Base.show(io::IO, path::LassoPath)
    prefix = isa(path.m, GeneralizedLinearModel) ? typeof(path.m.rr.d).name.name*" " : "" 
    println(io, prefix*"Lasso Solution Path ($(size(path.coefs, 2)) solutions for $(size(path.coefs, 1)) predictors):")
    Base.showarray(io, [path.λ path.pct_dev sum(path.coefs .!= 0, 1)']; header=false)
end

## FITTING (INCLUDING IRLS)

const MIN_DEV_FRAC_DIFF = 1e-5
const MAX_DEV_FRAC = 0.999

# Fits GLMs (outer and middle loops)
function StatsBase.fit{S<:GeneralizedLinearModel,T}(path::LassoPath{S,T}; verbose::Bool=false, irls_maxiter::Int=30,
                                                    cd_maxiter::Int=100000, cd_tol::Real=1e-7, irls_tol::Real=1e-7,
                                                    criterion=:obj, minStepFac::Real=eps())
    irls_maxiter >= 1 || error("irls_maxiter must be positive")
    0 < minStepFac < 1 || error("minStepFac must be in (0, 1)")
    criterion == :obj || criterion == :coef || error("criterion must be obj or coef")

    nulldev = path.nulldev
    λ = path.λ
    autoλ = path.autoλ
    Xnorm = path.Xnorm
    nλ = length(λ)
    m = path.m
    r = m.rr
    cd = m.pp
    cd.maxiter = cd_maxiter
    cd.tol = cd_tol

    X = cd.X
    α = cd.α
    coefs = Array(T, size(X, 2), nλ)
    b0s = zeros(T, nλ)
    oldcoef = SparseCoefficients{T}(size(X, 2))
    newcoef = SparseCoefficients{T}(size(X, 2))
    pct_dev = zeros(nλ)
    dev_ratio = convert(T, NaN)
    dev = convert(T, NaN)
    b0 = zero(T)
    scratchmu = Array(T, size(X, 1))
    eta = r.eta
    wrkresid = r.wrkresid
    objold = convert(T, Inf)

    if autoλ
        # No need to fit the first model
        coefs[:, 1] = zero(T)
        b0s[1] = path.nullb0
        i = 2
    else
        i = 1
    end

    dev = NaN
    while true # outer loop
        obj = convert(T, Inf)
        last_dev_ratio = dev_ratio
        curλ = λ[i]

        converged = false

        for iirls=1:irls_maxiter # middle loop
            copy!(oldcoef, newcoef)
            oldb0 = b0

            # Compute working response
            for j = 1:length(wrkresid)
                scratchmu[j] = eta[j] + wrkresid[j]
            end
            wrkwt = wrkwt!(r)

            # Run coordinate descent inner loop
            fit!(newcoef, update!(cd, newcoef, scratchmu, wrkwt), curλ, criterion)
            b0 = intercept(newcoef, cd)

            # Update GLM and get deviance
            dev = updatemu!(r, linpred!(scratchmu, X, newcoef, b0))

            # Compute Elastic Net objective
            objold = obj
            obj = dev/2 + curλ*P(α, newcoef)

            if obj > objold
                f = 1.0
                b0diff = b0 - oldb0
                while obj > objold
                    f /= 2.; f > minStepFac || error("step-halving failed at beta = $(newcoef)")
                    for icoef = 1:nnz(newcoef)
                        oldcoefval = icoef > nnz(oldcoef) ? zero(T) : oldcoef.coef[icoef]
                        newcoef.coef[icoef] = oldcoefval+f*(newcoef.coef[icoef] - oldcoefval)
                    end
                    b0 = oldb0+f*b0diff
                    dev = updatemu!(r, linpred!(scratchmu, X, newcoef, b0))
                    obj = dev/2 + curλ*P(α, newcoef)
                end
            end

            # Determine if we have converged
            if criterion == :obj
                converged = objold - obj < irls_tol*obj
            elseif criterion == :coef
                maxdelta = zero(T)
                Xssq = cd.Xssq
                weightsuminv = cd.weightsuminv
                for icoef = 1:nnz(newcoef)
                    oldcoefval = icoef > nnz(oldcoef) ? zero(T) : oldcoef.coef[icoef]
                    ipred = newcoef.coef2predictor[icoef]
                    maxdelta = max(maxdelta, abs2(oldcoefval - newcoef.coef[icoef])*Xssq[ipred]*weightsuminv)
                end

                converged = maxdelta < irls_tol
            end

            converged && break
        end
        converged || error("IRLS failed to converge in $irls_maxiter iterations at λ = $(curλ)")

        dev_ratio = dev/nulldev

        pct_dev[i] = 1 - dev_ratio
        coefs[:, i] = newcoef
        b0s[i] = b0

        # Test whether we should continue
        if i == nλ || (autoλ && last_dev_ratio - dev_ratio < MIN_DEV_FRAC_DIFF ||
                       pct_dev[i] > MAX_DEV_FRAC)
            break
        end

        i += 1
    end

    path.λ = path.λ[1:i]
    path.pct_dev = pct_dev[1:i]
    path.coefs = coefs[:, 1:i]
    path.b0 = b0s[1:i]
    if !isempty(Xnorm)
        scale!(Xnorm, path.coefs)
    end
end

# Fits linear models (just the outer loop)
function StatsBase.fit{S<:LinearModel,T}(path::LassoPath{S,T}; verbose::Bool=false, irls_maxiter::Int=30,
                                         cd_maxiter::Int=10000, cd_tol::Real=1e-7, irls_tol::Real=1e-7,
                                         criterion=:obj, minStepFac::Real=eps())
    irls_maxiter >= 1 || error("irls_maxiter must be positive")
    0 < minStepFac < 1 || error("minStepFac must be in (0, 1)")
    criterion == :obj || criterion == :coef || error("criterion must be obj or coef")

    nulldev = path.nulldev
    λ = path.λ
    autoλ = path.autoλ
    Xnorm = path.Xnorm
    nλ = length(λ)
    m = path.m
    r = m.rr
    cd = m.pp
    cd.maxiter = cd_maxiter
    cd.tol = cd_tol

    X = cd.X
    coefs = Array(T, size(X, 2), nλ)
    b0s = zeros(T, nλ)
    newcoef = SparseCoefficients{T}(size(X, 2))
    pct_dev = zeros(nλ)
    dev_ratio = convert(T, NaN)

    update!(cd, newcoef, r.y, r.wts)

    if autoλ
        # No need to fit the first model
        coefs[:, 1] = zero(T)
        b0s[1] = path.nullb0
        i = 2
    else
        i = 1
    end

    while true # outer loop
        last_dev_ratio = dev_ratio
        curλ = λ[i]

        # Run coordinate descent
        fit!(newcoef, cd, curλ, criterion)

        dev_ratio = cd.dev/nulldev
        pct_dev[i] = 1 - dev_ratio
        coefs[:, i] = newcoef
        b0s[i] = intercept(newcoef, cd)

        # Test whether we should continue
        if i == nλ || (autoλ && last_dev_ratio - dev_ratio < MIN_DEV_FRAC_DIFF ||
                       pct_dev[i] > MAX_DEV_FRAC)
            break
        end

        i += 1
    end

    path.λ = path.λ[1:i]
    path.pct_dev = pct_dev[1:i]
    path.coefs = coefs[:, 1:i]
    path.b0 = b0s[1:i]
    if !isempty(Xnorm)
        scale!(Xnorm, path.coefs)
    end
end

## MODEL CONSTRUCTION

function computeλ(Xy, λminratio, nλ)
    λmax = abs(Xy[1])
    for i = 1:length(Xy)
        x = abs(Xy[i])
        λmax = ifelse(x > λmax, x, λmax)
    end
    logλmax = log(λmax)
    λ = exp(linspace(logλmax, logλmax + log(λminratio), nλ))
end

function StatsBase.fit{T<:FloatingPoint,V<:FPVector}(::Type{LassoPath},
                                                     X::Matrix{T}, y::V, d::UnivariateDistribution=Normal(),
                                                     l::Link=canonicallink(d);
                                                     wts::Union(FPVector, Nothing)=ones(T, length(y)),
                                                     offset::V=similar(y, 0),
                                                     α::Number=one(eltype(y)), nλ::Int=100,
                                                     λminratio::Number=ifelse(size(X, 1) < size(X, 2), 0.01, 1e-4),
                                                     λ::Union(Vector,Nothing)=nothing, standardize::Bool=true,
                                                     intercept::Bool=true, naivealgorithm::Bool=true, dofit::Bool=true,
                                                     irls_tol::Real=1e-7, fitargs...)
    size(X, 1) == size(y, 1) || DimensionMismatch("number of rows in X and y must match")
    n = length(y)
    length(wts) == n || error("length(wts) = $(length(wts)) should be 0 or $n")

    # Standardize predictors if requested
    if standardize
        Xnorm = vec(std(X, 1, corrected=false))
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
    cd = naivealgorithm ? NaiveCoordinateDescent{T}(X, intercept, α, typemax(Int), 1e-7) :
                          CovarianceCoordinateDescent{T}(X, intercept, α, typemax(Int), 1e-7)

    # GLM response initialization
    autoλ = λ == nothing
    wts = eltype(wts) == T ? scale(wts, 1/sum(wts)) : scale!(convert(typeof(y), wts), 1/n)
    off = eltype(offset) == T ? copy(offset) : convert(Vector{T}, offset)

    if isa(d, Normal) && isa(l, IdentityLink)
        # Special no-IRLS case

        nullb0 = intercept ? mean(y, weights(wts)) : zero(T)
        nulldev = 0.0
        @simd for i = 1:length(y)
            @inbounds nulldev += abs2(y[i] - nullb0)*wts[i]
        end

        if autoλ
            # Find max λ
            if intercept
                yscratch = Array(T, length(y))
                @simd for i = 1:length(y)
                    @inbounds yscratch[i] = (y[i] - nullb0)*wts[i]
                end
            else
                yscratch = y.*wts
            end
            Xy = X'yscratch
            λ = computeλ(Xy, λminratio, nλ)
        else
            λ = convert(Vector{T}, λ)
        end

        # First y is just a placeholder here
        model = LinearModel(LmResp{typeof(y)}(y, off, wts, y), cd)
    else
        # Fit to find null deviance
        # Maybe we should use this GlmResp object?
        nullmodel = fit(GeneralizedLinearModel, ones(T, n, ifelse(intercept, 1, 0)), y, d, l; wts=wts, offset=offset, convTol=irls_tol)
        nulldev = deviance(nullmodel)

        if autoλ
            # Find max λ
            Xy = X'*broadcast!(*, nullmodel.rr.wrkresid, nullmodel.rr.wrkresid, nullmodel.rr.wrkwts)
            λ = computeλ(Xy, λminratio, nλ)
            @show nullb0
            nullb0 = intercept ? coef(nullmodel)[1] : zero(T)
        else
            λ = convert(Vector{T}, λ)
            nullb0 = zero(T)
        end

        mu = mustart(d, y, wts)
        eta = linkfun!(l, similar(mu), mu)
        if !isempty(off)
            subtract!(eta, off)
        end

        if length(GlmResp.parameters) == 3
            # a hack so this works with my hacked up working copy of GLM
            rr = GlmResp{typeof(y),typeof(d),typeof(l)}(y, d, l, eta, mu, offset, wts)
        else
            rr = GlmResp{typeof(y)}(y, d, l, eta, mu, offset, wts)
        end
        model = GeneralizedLinearModel(rr, cd, false)
    end

    # Fit path
    path = LassoPath{typeof(model),T}(model, nulldev, nullb0, λ, autoλ, Xnorm)
    dofit && fit(path; irls_tol=irls_tol, fitargs...)
    path
end


end
