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

## COEFFICIENT ITERATION IN SEQUENTIAL OR RANDOM ORDER

if VERSION >= v"0.4-dev+1915"
    immutable RandomCoefficientIterator
        rng::MersenneTwister
        rg::Base.Random.RangeGeneratorInt{Int,Uint}
        coeforder::Vector{Int}
    end
    const RANDOMIZE_DEFAULT = true
else
    immutable RandomCoefficientIterator
        RandomCoefficientIterator() = error("randomization not supported on Julia 0.3")
    end
    const RANDOMIZE_DEFAULT = false
end

RandomCoefficientIterator() = RandomCoefficientIterator(MersenneTwister(1337), Base.Random.RangeGenerator(1:2), Int[])

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

function addcoef(x::RandomCoefficientIterator, icoef::Int)
    push!(x.coeforder, icoef)
    RandomCoefficientIterator(x.rng, Base.Random.RangeGenerator(1:length(x.coeforder)), x.coeforder)
end
typealias CoefficientIterator Union(UnitRange{Int}, RandomCoefficientIterator)

addcoef(x::UnitRange{Int}, icoef::Int) = 1:length(x)+1

## COORDINATE DESCENT ROUTINES
S(z, γ) = abs(z) <= γ ? zero(z) : ifelse(z > 0, z - γ, z + γ)

function P{T}(α::T, β::SparseCoefficients{T})
    x = zero(T)
    @inbounds @simd for i = 1:nnz(β)
        x += (1 - α)/2*abs2(β.coef[i]) + α*abs(β.coef[i])
    end
    x
end

abstract CoordinateDescent{T} <: LinPred
type NaiveCoordinateDescent{T,S<:CoefficientIterator} <: CoordinateDescent{T}
    X::Matrix{T}                  # original design matrix
    μy::T                         # mean of y at current weights
    μX::Vector{T}                 # mean of X at current weights
    Xssq::Vector{T}               # weighted sum of squares of each column of X
    residuals::Vector{T}          # y - Xβ (unscaled with centered X)
    weights::Vector{T}            # weights for each observation
    oldy::Vector{T}               # old y vector (for updating residuals
                                  # without matrix multiplication)
    weightsum::T                  # sum(weights)
    coefitr::S                    # coefficient iterator
    dev::T                        # last deviance
    intercept::Bool               # whether an intercept should be fitted
    α::T                          # elastic net parameter
    maxiter::Int                  # maximum number of iterations
    maxncoef::Int                 # maximum number of coefficients
    tol::T                        # tolerance

    NaiveCoordinateDescent{T,S}(X::Matrix{T}, intercept::Bool, α::T, maxncoef::Int, tol::T, coefitr::S)  =
        new(X, zero(T), zeros(T, maxncoef), zeros(T, maxncoef), Array(T, size(X, 1)), Array(T, size(X, 1)),
            Array(T, size(X, 1)), convert(T, NaN), coefitr, convert(T, NaN), intercept, α, typemax(Int),
            maxncoef, tol)
end

# Compute μX and Xssq for given predictor
function computeμ!{T}(cd::NaiveCoordinateDescent{T}, icoef::Int, ipred::Int)
    X = cd.X
    Xssq = cd.Xssq
    weights = cd.weights
    @inbounds if cd.intercept
        # Update μX
        μX = cd.μX
        μ = zero(T)
        @simd for i = 1:size(X, 1)
            μ += X[i, ipred]*weights[i]
        end
        μ /= cd.weightsum
        μX[icoef] = μ

        # Update Xssq
        ws = zero(T)
        @simd for i = 1:size(X, 1)
            ws += abs2(X[i, ipred] - μ)*weights[i]
        end
        Xssq[icoef] = ws
    else
        ws = zero(T)
        @simd for i = 1:size(X, 1)
            ws += abs2(X[i, ipred])*weights[i]
        end
        Xssq[icoef] = ws
    end
    cd
end

# Updates CoordinateDescent object with (possibly) new y vector and
# weights
function update!{T}(cd::NaiveCoordinateDescent{T}, coef::SparseCoefficients{T}, y::Vector{T}, wt::Vector{T})
    residuals = cd.residuals
    X = cd.X
    weights = copy!(cd.weights, wt)
    weightsum = cd.weightsum = sum(weights)
    weightsuminv = inv(weightsum)
    oldy = cd.oldy

    # Update residuals without recomputing X*coef
    if nnz(coef) == 0
        copy!(residuals, y)
        copy!(oldy, y)
    else
        @inbounds @simd for i = 1:length(y)
            residuals[i] += y[i] - oldy[i]
            oldy[i] = y[i]
        end
    end

    if cd.intercept
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
        nulldev = zero(T)
        @simd for i = 1:length(y)
            nulldev += abs2(y[i])*weights[i]
        end
    end
    for icoef = 1:nnz(coef)
        computeμ!(cd, icoef, coef.coef2predictor[icoef])
    end
    cd
end

if VERSION < v"0.4.0-dev+707"
    # no-op inline macro for old Julia
    macro inline(x)
        esc(x)
    end
end

@inline function update_residuals!(X, residuals, μ, coefdiff, ipred)
    @simd for i = 1:size(X, 1)
        @inbounds residuals[i] += coefdiff*(X[i, ipred] - μ)
    end
end

# Performs the cycle of all predictors
function cycle!{T}(coef::SparseCoefficients{T}, cd::NaiveCoordinateDescent{T}, λ::T, α::T, all::Bool)
    X = cd.X
    residuals = cd.residuals
    weights = cd.weights
    Xssq = cd.Xssq
    μX = cd.μX
    coefitr = cd.coefitr

    maxdelta = zero(T)
    @inbounds if all
        # Use all variables for first and last iterations
        for ipred = 1:size(X, 2)
            icoef = coef.predictor2coef[ipred]

            v = zero(T)
            @simd for i = 1:size(X, 1)
                @inbounds v += X[i, ipred]*residuals[i]*weights[i]
            end

            icoef = coef.predictor2coef[ipred]
            if icoef != 0
                oldcoef = coef.coef[icoef]
                v += Xssq[icoef]*oldcoef
            else
                # Adding a new variable to the model
                abs(v) < λ*α && continue
                oldcoef = zero(T)
                length(coef.coef) > cd.maxncoef &&
                    error("maximum number of coefficients $(cd.maxncoef) exceeded at λ = $λ")
                icoef = addcoef!(coef, ipred)
                cd.coefitr = addcoef(cd.coefitr, icoef)
                computeμ!(cd, icoef, ipred)
            end
            newcoef = S(v, λ*α)/(Xssq[icoef] + λ*(1 - α))

            if oldcoef != newcoef
                coef.coef[icoef] = newcoef
                update_residuals!(X, residuals, μX[icoef], oldcoef - newcoef, ipred)
                # println("1 ipred = ", ipred, " diff = ", oldcoef - newcoef, " sq = ", Xssq[ipred])
                maxdelta = max(maxdelta, abs2(oldcoef - newcoef)*Xssq[icoef])
            end
        end
    else
        for icoef = cd.coefitr
            oldcoef = coef.coef[icoef]
            oldcoef == 0 && continue
            ipred = coef.coef2predictor[icoef]

            v = Xssq[icoef]*oldcoef
            @simd for i = 1:size(X, 1)
                @inbounds v += X[i, ipred]*residuals[i]*weights[i]
            end
            newcoef = S(v, λ*α)/(Xssq[icoef] + λ*(1 - α))

            if oldcoef != newcoef
                coef.coef[icoef] = newcoef
                update_residuals!(X, residuals, μX[icoef], oldcoef - newcoef, ipred)
                # println("2 ipred = ", ipred, " diff = ", oldcoef - newcoef, " sq = ", Xssq[ipred])
                maxdelta = max(maxdelta, abs2(oldcoef - newcoef)*Xssq[icoef])
            end
        end
    end
    maxdelta#/cd.weightsum
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

type CovarianceCoordinateDescent{T,S<:CoefficientIterator} <: CoordinateDescent{T}
    X::Matrix{T}                  # original design matrix
    μy::T                         # mean of y at current weights
    μX::Vector{T}                 # mean of X at current weights
    yty::T                        # y'y (scaled by weights)
    Xty::Vector{T}                # X'y (scaled by weights)
    Xssq::Vector{T}               # weighted sum of squares of each column of X
    XtX::Matrix{T}                # X'X (scaled by weights, in order of coefficients)
    tmp::Vector{T}                # scratch used when computing X'X
    weights::Vector{T}            # weights for each observation
    weightsum::T                  # sum(weights)
    coefitr::S                    # coefficient iterator
    dev::T                        # last deviance
    intercept::Bool               # whether an intercept should be fitted
    α::T                          # elastic net parameter
    maxiter::Int                  # maximum number of iterations
    maxncoef::Int                 # maximum number of coefficients
    tol::T                        # tolerance

    function CovarianceCoordinateDescent{T}(X::Matrix{T}, intercept::Bool, α::T, maxncoef::Int, tol::T, coefiter::S)
        new(X, zero(T), zeros(T, size(X, 2)), convert(T, NaN), Array(T, size(X, 2)),
            Array(T, size(X, 2)), Array(T, maxncoef, size(X, 2)), Array(T, size(X, 1)),
            Array(T, size(X, 1)), convert(T, NaN), coefiter, convert(T, NaN), intercept, α,
            typemax(Int), maxncoef, tol)
    end
end

# Compute XtX
# Equivalent to (X .- μX)'(weights.*(X[:, icoef] - μX[icoef]))
#
# This must be called for all icoef < icoef before it may be called
# for a given icoef
function computeXtX!{T}(cd::CovarianceCoordinateDescent{T}, coef::SparseCoefficients{T}, icoef::Int, ipred::Int)
    X = cd.X
    tmp = cd.tmp
    weights = cd.weights
    XtX = cd.XtX
    μX = cd.μX

    μ = μX[ipred]
    @simd for i = 1:size(cd.X, 1)
        @inbounds tmp[i] = (X[i, ipred] - μ)*weights[i]
    end

    @inbounds for jpred = 1:size(X, 2)
        if jpred == ipred
            XtX[icoef, jpred] = cd.Xssq[jpred]
        else
            jcoef = coef.predictor2coef[jpred]
            if 0 < jcoef < icoef
                XtX[icoef, jpred] = XtX[jcoef, ipred]
            else
                s = zero(T)
                μ = μX[jpred]
                @simd for i = 1:size(X, 1)
                    s += tmp[i]*(X[i, jpred] - μ)
                end
                XtX[icoef, jpred] = s
            end
        end
    end
    cd
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
    weightsum = cd.weightsum = sum(weights)
    weightsuminv = inv(weightsum)
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
            ws = zero(T)
            v = zero(T)
            @inbounds @simd for i = 1:size(X, 1)
                ws += abs2(X[i, j] - μ)*weights[i]
                v += (y[i] - μy)*(X[i, j] - μ)*weights[i]
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
            @inbounds @simd for i = 1:size(X, 1)
                ws += abs2(X[i, j])*weights[i]
                v += y[i]*X[i, j]*weights[i]
            end
            Xssq[j] = ws
            Xty[j] = v
        end
    end

    for icoef = 1:nnz(coef)
        computeXtX!(cd, coef, icoef, coef.coef2predictor[icoef])
    end
    cd.yty = yty

    cd
end

function compute_gradient(XtX, coef, ipred)
    s = 0.0
    @simd for jcoef = 1:nnz(coef)
        @inbounds s += XtX[jcoef, ipred]*coef.coef[jcoef]
    end
    s
end

# Performs the cycle of all predictors
function cycle!{T}(coef::SparseCoefficients{T}, cd::CovarianceCoordinateDescent{T}, λ::T, α::T, all::Bool)
    Xty = cd.Xty
    XtX = cd.XtX
    Xssq = cd.Xssq

    maxdelta = 0.0
    if all
        @inbounds for ipred = 1:length(Xty)
            # Use all variables for first and last iterations
            s = Xty[ipred] - compute_gradient(XtX, coef, ipred)

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
                    # Adding a new variable to the model
                    length(coef.coef) > cd.maxncoef &&
                        error("maximum number of coefficients $(cd.maxncoef) exceeded at λ = $λ")
                    icoef = addcoef!(coef, ipred)
                    cd.coefitr = addcoef(cd.coefitr, icoef)

                    # Compute cross-product with predictors
                    computeXtX!(cd, coef, icoef, ipred)
                end
                maxdelta = max(maxdelta, abs2(oldcoef - newcoef)*Xssq[ipred])
                coef.coef[icoef] = newcoef
            end
            # println("s $j => $s, den = $((XtX[j, j] + λ*(1 - α)))")
            # println("$j => $(coef[j])")
        end
    else
        @inbounds for icoef = cd.coefitr
            ipred = coef.coef2predictor[icoef]
            oldcoef = coef.coef[icoef]
            oldcoef == 0 && continue
            s = Xty[ipred] + XtX[icoef, ipred]*oldcoef - compute_gradient(XtX, coef, ipred)
            newcoef = coef.coef[icoef] = S(s, λ*α)/(Xssq[ipred] + λ*(1 - α))
            maxdelta = max(maxdelta, abs2(oldcoef - newcoef)*Xssq[ipred])
        end
    end
    maxdelta#/cd.weightsum
end

# y'y - 2β'X'y + β'X'Xβ
function ssr{T}(coef::SparseCoefficients{T}, cd::CovarianceCoordinateDescent{T})
    XtX = cd.XtX
    Xty = cd.Xty
    v = cd.yty
    @inbounds for icoef = 1:nnz(coef)
        ipred = coef.coef2predictor[icoef]
        s = -2*Xty[ipred] + compute_gradient(XtX, coef, ipred)
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
    b0 = intercept(coef, cd)

    iter = 0
    for iter = 1:maxiter
        oldb0 = b0
        maxdelta = cycle!(coef, cd, λ, α, converged)
        b0 = intercept(coef, cd)
        maxdelta = max(maxdelta, abs2(oldb0 - b0)*cd.weightsum)
        # println("b0diff = ", oldb0 - b0, " weightsum = ", cd.weightsum)

        # Test for convergence
        prev_converged = converged
        if criterion == :obj
            objold = obj
            dev = ssr(coef, cd)
            obj = dev/2 + λ*P(α, coef)
            converged = objold - obj < tol*obj
        elseif criterion == :coef
            converged = maxdelta < tol
        end

        # Require two converging steps to return. After the first, we
        # will iterate through all variables
        prev_converged && converged && break
    end

    (!prev_converged || !converged) &&
        error("coordinate descent failed to converge in $maxiter iterations at λ = $λ")

    cd.dev = criterion == :coef ? ssr(coef, cd) : dev
    iter
end

intercept{T}(coef::SparseCoefficients{T}, cd::CovarianceCoordinateDescent{T}) =
    cd.intercept ? cd.μy - dot(cd.μX, coef) : zero(T)
function intercept{T}(coef::SparseCoefficients{T}, cd::NaiveCoordinateDescent{T})
    !cd.intercept && return zero(T)
    μX = cd.μX
    v = cd.μy
    for i = 1:length(coef.coef)
        v -= μX[i]*coef.coef[i]
    end
    v
end

function linpred!{T}(mu::Vector{T}, cd::NaiveCoordinateDescent{T}, coef::SparseCoefficients{T}, offset, b0::T)
    oldy = cd.oldy
    residuals = cd.residuals
    @simd for i = 1:length(mu)
        @inbounds mu[i] = oldy[i] - residuals[i]
    end
    mu
end

function linpred!{T}(mu::Vector{T}, cd::CovarianceCoordinateDescent{T}, coef::SparseCoefficients{T}, offset, b0::T)
    A_mul_B!(mu, cd.X, coef)
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
    Base.showarray(io, [path.λ path.pct_dev sum(path.coefs .!= 0, 1)']; header=false)
end

## FITTING (INCLUDING IRLS)

const MIN_DEV_FRAC_DIFF = 1e-5
const MAX_DEV_FRAC = 0.999

function addcoefs!(coefs::SparseMatrixCSC, newcoef::SparseCoefficients, i::Int)
    n = nnz(coefs)
    nzval = coefs.nzval
    rowval = coefs.rowval
    resize!(nzval, n+length(newcoef.coef))
    resize!(rowval, n+length(newcoef.coef))
    @inbounds for ipred = 1:length(newcoef.predictor2coef)
        icoef = newcoef.predictor2coef[ipred]
        if icoef != 0
            n += 1
            nzval[n] = newcoef.coef[icoef]
            rowval[n] = ipred
        end
    end
    coefs.colptr[i+1:end] = length(coefs.nzval)+1
end

function wrkresp!(out, x, y, offset)
    if isempty(offset)
        @simd for i = 1:length(out)
            @inbounds out[i] = x[i] + y[i]
        end
    else
        @simd for i = 1:length(out)
            @inbounds out[i] = x[i] + y[i] - offset[i]
        end
    end
    out
end

# Fits GLMs (outer and middle loops)
function StatsBase.fit{S<:GeneralizedLinearModel,T}(path::LassoPath{S,T}; verbose::Bool=false, irls_maxiter::Int=30,
                                                    cd_maxiter::Int=100000, cd_tol::Real=1e-7, irls_tol::Real=1e-7,
                                                    criterion=:coef, minStepFac::Real=eps())
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

    if criterion == :coef
        cd_tol *= path.nulldev/2
        irls_tol *= path.nulldev/2
    end

    X = cd.X
    α = cd.α
    coefs = spzeros(T, size(X, 2), nλ)
    b0s = zeros(T, nλ)
    oldcoef = SparseCoefficients{T}(size(X, 2))
    newcoef = SparseCoefficients{T}(size(X, 2))
    pct_dev = zeros(nλ)
    dev_ratio = convert(T, NaN)
    dev = convert(T, NaN)
    b0 = zero(T)
    scratchmu = Array(T, size(X, 1))
    offset = r.offset
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
    niter = 0
    if nλ == 0
        i = 0
    else
        while true # outer loop
            obj = convert(T, Inf)
            last_dev_ratio = dev_ratio
            curλ = λ[i]
            # println()
            # println("λ = $curλ")

            converged = false

            for iirls=1:irls_maxiter # middle loop
                copy!(oldcoef, newcoef)
                oldb0 = b0

                # Compute working response
                wrkresp!(scratchmu, eta, wrkresid, offset)
                wrkwt = wrkwt!(r)

                # Run coordinate descent inner loop
                niter += fit!(newcoef, update!(cd, newcoef, scratchmu, wrkwt), curλ, criterion)
                b0 = intercept(newcoef, cd)

                # Update GLM and get deviance
                dev = updatemu!(r, linpred!(scratchmu, cd, newcoef, offset, b0))

                # Compute Elastic Net objective
                objold = obj
                obj = dev/2 + curλ*P(α, newcoef)

                if obj > objold + length(scratchmu)*eps(objold)
                    f = 1.0
                    b0diff = b0 - oldb0
                    while obj > objold
                        f /= 2.; f > minStepFac || error("step-halving failed at beta = $(newcoef)")
                        for icoef = 1:nnz(newcoef)
                            oldcoefval = icoef > nnz(oldcoef) ? zero(T) : oldcoef.coef[icoef]
                            newcoef.coef[icoef] = oldcoefval+f*(newcoef.coef[icoef] - oldcoefval)
                        end
                        b0 = oldb0+f*b0diff
                        dev = updatemu!(r, linpred!(scratchmu, cd, newcoef, offset, b0))
                        obj = dev/2 + curλ*P(α, newcoef)
                    end
                end

                # Determine if we have converged
                if criterion == :obj
                    converged = objold - obj < irls_tol*obj
                elseif criterion == :coef
                    maxdelta = zero(T)
                    Xssq = cd.Xssq
                    converged = abs2(oldb0 - b0)*cd.weightsum < irls_tol
                    if converged
                        for icoef = 1:nnz(newcoef)
                            oldcoefval = icoef > nnz(oldcoef) ? zero(T) : oldcoef.coef[icoef]
                            j = isa(cd, NaiveCoordinateDescent) ? icoef : newcoef.coef2predictor[icoef]
                            if abs2(oldcoefval - newcoef.coef[icoef])*Xssq[j] > irls_tol
                                converged = false
                                break
                            end
                        end
                    end
                end

                converged && break
            end
            converged || error("IRLS failed to converge in $irls_maxiter iterations at λ = $(curλ)")

            dev_ratio = dev/nulldev

            pct_dev[i] = 1 - dev_ratio
            addcoefs!(coefs, newcoef, i)
            b0s[i] = b0

            # Test whether we should continue
            if i == nλ || (autoλ && last_dev_ratio - dev_ratio < MIN_DEV_FRAC_DIFF ||
                           pct_dev[i] > MAX_DEV_FRAC)
                break
            end

            i += 1
        end
    end

    path.λ = path.λ[1:i]
    path.pct_dev = pct_dev[1:i]
    path.coefs = coefs[:, 1:i]
    path.b0 = b0s[1:i]
    path.niter = niter
    if !isempty(Xnorm)
        scale!(Xnorm, path.coefs)
    end
end

# Fits linear models (just the outer loop)
function StatsBase.fit{S<:LinearModel,T}(path::LassoPath{S,T}; verbose::Bool=false,
                                         cd_maxiter::Int=10000, cd_tol::Real=1e-7, irls_tol::Real=1e-7,
                                         criterion=:coef, minStepFac::Real=eps())
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
    coefs = spzeros(T, size(X, 2), nλ)
    b0s = zeros(T, nλ)
    newcoef = SparseCoefficients{T}(size(X, 2))
    pct_dev = zeros(nλ)
    dev_ratio = convert(T, NaN)
    niter = 0

    update!(cd, newcoef, r.mu, r.wts)

    if autoλ
        # No need to fit the first model
        b0s[1] = path.nullb0
        i = 2
    else
        i = 1
    end

    while true # outer loop
        last_dev_ratio = dev_ratio
        curλ = λ[i]

        # Run coordinate descent
        niter += fit!(newcoef, cd, curλ, criterion)

        dev_ratio = cd.dev/nulldev
        pct_dev[i] = 1 - dev_ratio
        addcoefs!(coefs, newcoef, i)
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
    path.niter = niter
    if !isempty(Xnorm)
        scale!(Xnorm, path.coefs)
    end
end

## MODEL CONSTRUCTION

function computeλ(Xy, λminratio, α, nλ)
    λmax = abs(Xy[1])
    for i = 1:length(Xy)
        x = abs(Xy[i])
        λmax = ifelse(x > λmax, x, λmax)
    end
    λmax /= α
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
                                                     irls_tol::Real=1e-7, randomize::Bool=RANDOMIZE_DEFAULT,
                                                     maxncoef::Int=min(size(X, 2), 2*size(X, 1)), fitargs...)
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
    coefitr = randomize ? RandomCoefficientIterator() : (1:0)
    cd = naivealgorithm ? NaiveCoordinateDescent{T,typeof(coefitr)}(X, intercept, α, maxncoef, 1e-7, coefitr) :
                          CovarianceCoordinateDescent{T,typeof(coefitr)}(X, intercept, α, maxncoef, 1e-7, coefitr)

    # GLM response initialization
    autoλ = λ == nothing
    wts = eltype(wts) == T ? scale(wts, 1/sum(wts)) : scale!(convert(typeof(y), wts), 1/n)
    off = eltype(offset) == T ? offset : convert(Vector{T}, offset)

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
            λ = computeλ(Xy, λminratio, α, nλ)
        else
            λ = convert(Vector{T}, λ)
        end

        # First y is just a placeholder here
        model = LinearModel(LmResp{typeof(y)}(mu, off, wts, y), cd)
    else
        # Fit to find null deviance
        # Maybe we should use this GlmResp object?
        nullmodel = fit(GeneralizedLinearModel, ones(T, n, ifelse(intercept, 1, 0)), y, d, l;
                        wts=wts, offset=offset, convTol=irls_tol)
        nulldev = deviance(nullmodel)

        if autoλ
            # Find max λ
            Xy = X'*broadcast!(*, nullmodel.rr.wrkresid, nullmodel.rr.wrkresid, nullmodel.rr.wrkwts)
            λ = computeλ(Xy, λminratio, α, nλ)
            nullb0 = intercept ? coef(nullmodel)[1] : zero(T)
        else
            λ = convert(Vector{T}, λ)
            nullb0 = zero(T)
        end

        mu = mustart(d, y, wts)
        eta = linkfun!(l, similar(mu), mu)
        if !isempty(off)
            @simd for i = 1:length(eta)
                @inbounds eta[i] -= off[i]
            end
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
