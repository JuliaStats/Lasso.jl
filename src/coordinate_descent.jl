# Soft threshold function
S(z, γ) = abs(z) <= γ ? zero(z) : ifelse(z > 0, z - γ, z + γ)

# Elastic net penalty with parameter α and given coefficients
function P(α::T, β::SparseCoefficients{T}, ω::Nothing) where T
    x = zero(T)
    @inbounds @simd for i = 1:nnz(β)
        x += (1 - α)/2*abs2(β.coef[i]) + α*abs(β.coef[i])
    end
    x
end

function P(α::T, β::SparseCoefficients{T}, ω::Vector{T}) where T
    x = zero(T)
    @inbounds @simd for i = 1:nnz(β)
        x += ω[β.coef2predictor[i]] * ((1 - α)/2*abs2(β.coef[i]) + α*abs(β.coef[i]))
    end
    x
end

abstract type CoordinateDescent{T,Intercept,M<:AbstractMatrix} <: LinPred end

mutable struct NaiveCoordinateDescent{T,Intercept,M<:AbstractMatrix,S<:CoefficientIterator,W<:Union{Vector,Nothing}} <: CoordinateDescent{T,Intercept,M}
    X::M                          # original design matrix
    μy::T                         # mean of y at current weights
    μX::Vector{T}                 # mean of X at current weights (in predictor order)
    Xssq::Vector{T}               # weighted sum of squares of each column of X (in coefficient order)
    residuals::Vector{T}          # y - Xβ (unscaled with centered X)
    residualoffset::T             # offset of residuals (used only when X is sparse)
    weights::Vector{T}            # weights for each observation
    oldy::Vector{T}               # old y vector (for updating residuals without matrix multiplication)
    weightsum::T                  # sum(weights)
    coefitr::S                    # coefficient iterator
    dev::T                        # last deviance
    α::T                          # elastic net parameter
    maxiter::Int                  # maximum number of iterations
    maxncoef::Int                 # maximum number of coefficients
    tol::T                        # tolerance
    ω::W                          # coefficient-specific penalty weights

    NaiveCoordinateDescent{T,Intercept,M,S,W}(X::M, α::Real, maxncoef::Int, tol::Real, coefitr::S, ω::Union{Vector{T},Nothing}) where {T,Intercept,M,S,W} =
        new(X, zero(T), zeros(T, size(X, 2)), zeros(T, maxncoef), Vector{T}(undef, size(X, 1)), zero(T),
            Vector{T}(undef, size(X, 1)), Vector{T}(undef, size(X, 1)), convert(T, NaN), coefitr, convert(T, NaN),
            α, typemax(Int), maxncoef, tol, ω)
end

# Compute μX for all predictors
function computeμX!(cd::CoordinateDescent{T}) where T
    @extractfields cd X μX weights weightsum
    for ipred = 1:size(X, 2)
        μ = zero(T)
        @simd for i = 1:size(X, 1)
            @inbounds μ += X[i, ipred]*weights[i]
        end
        μX[ipred] = μ/weightsum
    end
    cd
end

function computeμX!(cd::CoordinateDescent{T,Intercept,M}) where {T,Intercept,M<:SparseMatrixCSC}
    @extractfields cd X μX weights weightsum
    @extractfields X rowval nzval colptr
    for ipred = 1:size(X, 2)
        μ = zero(T)
        @simd for i = colptr[ipred]:colptr[ipred+1]-1
            row = rowval[i]
            @inbounds μ += nzval[i]*weights[row]
        end
        μX[ipred] = μ/weightsum
    end
    cd
end

# Compute Xssq for given predictor, with mean subtracted if there is
# an intercept
function computeXssq(cd::NaiveCoordinateDescent{T,Intercept}, ipred::Int) where {T,Intercept}
    @extractfields cd X weights
    μ = Intercept ? cd.μX[ipred] : zero(T)
    ssq = zero(T)
    @simd for i = 1:size(X, 1)
        @inbounds ssq += abs2(X[i, ipred] - μ)*weights[i]
    end
    ssq
end

function computeXssq(cd::NaiveCoordinateDescent{T,Intercept,M}, ipred::Int) where {T,Intercept,M<:SparseMatrixCSC}
    @extractfields cd X weights
    @extractfields X rowval nzval colptr
    μ = Intercept ? cd.μX[ipred] : zero(T)
    ssq = zero(T)
    zeroweightsum = cd.weightsum
    @inbounds @simd for i = colptr[ipred]:colptr[ipred+1]-1
        row = rowval[i]
        ssq += abs2(nzval[i] - μ)*weights[row]
        zeroweightsum -= weights[row]
    end
    ssq + zeroweightsum*abs2(μ)
end

# Updates CoordinateDescent object with (possibly) new y vector and
# weights
function update!(cd::NaiveCoordinateDescent{T,Intercept}, coef::SparseCoefficients{T},
                              y::Vector{T}, wt::Vector{T}) where {T,Intercept}
    @extractfields cd residuals X Xssq weights oldy
    copyto!(weights, wt)
    weightsum = cd.weightsum = sum(weights)
    weightsuminv = inv(weightsum)

    # Update residuals without recomputing X*coef
    if nnz(coef) == 0
        copyto!(residuals, y)
        copyto!(oldy, y)
    else
        cd.residualoffset = 0
        @inbounds @simd for i = 1:length(y)
            residuals[i] += y[i] - oldy[i]
            oldy[i] = y[i]
        end
    end

    if Intercept
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

        computeμX!(cd)
    end

    for icoef = 1:nnz(coef)
        Xssq[icoef] = computeXssq(cd, coef.coef2predictor[icoef])
    end
    cd
end

# Offset of each residual. This is used only for sparse matrices with
# an intercept, so that we can update only residuals for which a
# changed coefficient is non-zero instead of updating all of them. In
# the dense case, we need to update all coefficients anyway, so this
# strategy is unneeded.
residualoffset(cd::NaiveCoordinateDescent{T}) where {T} = zero(T)
residualoffset(cd::NaiveCoordinateDescent{T,true,M}) where {T,M<:SparseMatrixCSC} = cd.residualoffset

# Compute the gradient term (first term of RHS of eq. 8)
@inline function compute_grad(::NaiveCoordinateDescent{T}, X::AbstractMatrix{T},
                              residuals::Vector{T}, weights::Vector{T}, ipred::Int) where T
    v = zero(T)
    @simd for i = 1:size(X, 1)
        @inbounds v += X[i, ipred]*residuals[i]*weights[i]
    end
    v
end

@inline function compute_grad(cd::NaiveCoordinateDescent{T}, X::SparseMatrixCSC{T},
                              residuals::Vector{T}, weights::Vector{T}, ipred::Int) where T
    @extractfields X rowval nzval colptr
    @inbounds v = residualoffset(cd)*cd.weightsum*cd.μX[ipred]
    @inbounds @simd for i = colptr[ipred]:colptr[ipred+1]-1
        row = rowval[i]
        v += nzval[i]*residuals[row]*weights[row]
    end
    v
end

# Update coefficient and residuals, returning scaled squared difference
@inline function update_coef!(cd::NaiveCoordinateDescent{T,Intercept},
                              coef::SparseCoefficients{T},
                              newcoef::T, icoef::Int, ipred::Int) where {T,Intercept}
    coefdiff = coef.coef[icoef] - newcoef
    if coefdiff != 0
        coef.coef[icoef] = newcoef
        μ = Intercept ? cd.μX[ipred] : zero(T)

        @extractfields cd X residuals
        @simd for i = 1:size(X, 1)
            @inbounds residuals[i] += coefdiff*(X[i, ipred] - μ)
        end
        abs2(coefdiff)*cd.Xssq[icoef]
    else
        zero(T)
    end
end

@inline function update_coef!(cd::NaiveCoordinateDescent{T,Intercept,M},
                              coef::SparseCoefficients{T},
                              newcoef::T, icoef::Int, ipred::Int) where {T,Intercept,M<:SparseMatrixCSC}
    coefdiff = coef.coef[icoef] - newcoef
    if coefdiff != 0
        coef.coef[icoef] = newcoef
        @extractfields cd X residuals
        @extractfields X rowval nzval colptr
        if Intercept
            cd.residualoffset -= coefdiff*cd.μX[ipred]
        end
        @inbounds @simd for i = colptr[ipred]:colptr[ipred+1]-1
            row = rowval[i]
            residuals[row] += coefdiff*nzval[i]
        end
        abs2(coefdiff)*cd.Xssq[icoef]
    else
        zero(T)
    end
end

λω(λ,ω::Nothing,ipred::Int) = λ
λω(λ,ω::Vector,ipred::Int) = λ*ω[ipred]

# Performs the cycle of all predictors
function cycle!(coef::SparseCoefficients{T}, cd::NaiveCoordinateDescent{T}, λ::T, all::Bool) where T
    @extractfields cd residuals X weights Xssq α ω

    maxdelta = zero(T)
    @inbounds if all
        # Use all predictors for first and last iterations
        for ipred = 1:size(X, 2)
            v = compute_grad(cd, X, residuals, weights, ipred)
            λωj = λω(λ,ω,ipred)

            icoef = coef.predictor2coef[ipred]
            if icoef != 0
                oldcoef = coef.coef[icoef]
                v += Xssq[icoef]*oldcoef
            else
                # Adding a new variable to the model
                abs(v) < λωj*α && continue
                oldcoef = zero(T)
                nnz(coef) > cd.maxncoef &&
                    error("maximum number of coefficients $(cd.maxncoef) exceeded at λ = $λ (λωj=$λωj)")
                icoef = addcoef!(coef, ipred)
                cd.coefitr = addcoef(cd.coefitr, icoef)
                Xssq[icoef] = computeXssq(cd, ipred)
            end
            newcoef = S(v, λωj*α)/(Xssq[icoef] + λωj*(1 - α))

            maxdelta = max(maxdelta, update_coef!(cd, coef, newcoef, icoef, ipred))
        end
    else
        # Iterate over only the predictors already in the model
        for icoef = cd.coefitr
            oldcoef = coef.coef[icoef]
            oldcoef == 0 && continue
            ipred = coef.coef2predictor[icoef]

            v = Xssq[icoef]*oldcoef + compute_grad(cd, X, residuals, weights, ipred)
            λωj = λω(λ,ω,ipred)
            newcoef = S(v, λωj*α)/(Xssq[icoef] + λωj*(1 - α))

            maxdelta = max(maxdelta, update_coef!(cd, coef, newcoef, icoef, ipred))
        end
    end
    maxdelta
end

# Sum of squared residuals. Residuals are always up to date
function ssr(coef::SparseCoefficients{T}, cd::NaiveCoordinateDescent{T}) where T
    @extractfields cd residuals weights
    roffset = residualoffset(cd)
    s = zero(T)
    @simd for i = 1:length(residuals)
        @inbounds s += abs2(residuals[i] + roffset)*weights[i]
    end
    s
end

# Does the linear predictor have an intercept?
hasintercept(cd::CoordinateDescent{T,false}) where {T} = false
hasintercept(cd::CoordinateDescent{T,true}) where {T} = true

# Value of the intercept
intercept(coef::SparseCoefficients{T}, cd::CoordinateDescent{T,false}) where {T} = zero(T)
function intercept(coef::SparseCoefficients{T}, cd::NaiveCoordinateDescent{T,true}) where T
    μX = cd.μX
    v = cd.μy
    for icoef = 1:nnz(coef)
        v -= μX[coef.coef2predictor[icoef]]*coef.coef[icoef]
    end
    v
end

# Value of the linear predictor
function linpred!(mu::Vector{T}, cd::NaiveCoordinateDescent{T}, coef::SparseCoefficients{T}, b0::T) where T
    @extractfields cd oldy residuals
    roffset = residualoffset(cd)
    @simd for i = 1:length(mu)
        @inbounds mu[i] = oldy[i] - (residuals[i] + roffset)
    end
    mu
end

mutable struct CovarianceCoordinateDescent{T,Intercept,M<:AbstractMatrix,S<:CoefficientIterator,W<:Union{Vector,Nothing}} <: CoordinateDescent{T,Intercept,M}
    X::M                          # original design matrix
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
    α::T                          # elastic net parameter
    maxiter::Int                  # maximum number of iterations
    maxncoef::Int                 # maximum number of coefficients
    tol::T                        # tolerance
    ω::W                          # coefficient-specific penalty weights

    function CovarianceCoordinateDescent{T,Intercept,M,S,W}(X::M, α::Real, maxncoef::Int, tol::Real, coefiter::S, ω::Union{Vector,Nothing}) where {T,Intercept,M,S,W}
        new(X, zero(T), zeros(T, size(X, 2)), convert(T, NaN), Vector{T}(undef, size(X, 2)),
            Vector{T}(undef, size(X, 2)), Matrix{T}(undef, maxncoef, size(X, 2)), Vector{T}(undef, size(X, 1)),
            Vector{T}(undef, size(X, 1)), convert(T, NaN), coefiter, convert(T, NaN), α,
            typemax(Int), maxncoef, tol, ω)
    end
end

# Compute y'y
function computeyty(cd::CovarianceCoordinateDescent{T,Intercept,M}, y::Vector{T}) where {T,Intercept,M}
    @extractfields cd X weights
    μy = Intercept ? cd.μy : zero(T)
    yty = zero(T)
    @inbounds @simd for i = 1:length(y)
        yty += abs2(y[i] - μy)*weights[i]
    end
    yty
end

# Compute Xssq and Xty for given predictor, with mean subtracted if
# there is an intercept.
function computeXssqXty(cd::CovarianceCoordinateDescent{T,Intercept}, y::Vector{T}, ipred::Int) where {T,Intercept}
    @extractfields cd X μy weights
    μ = Intercept ? cd.μX[ipred] : zero(T)
    ssq = zero(T)
    ty = zero(T)
    @inbounds @simd for i = 1:size(X, 1)
        ssq += abs2(X[i, ipred] - μ)*weights[i]
        ty += (y[i] - μy)*(X[i, ipred] - μ)*weights[i]
    end
    (ssq, ty)
end

function computeXssqXty(cd::CovarianceCoordinateDescent{T,Intercept,M}, y::Vector{T}, ipred::Int) where {T,Intercept,M<:SparseMatrixCSC}
    @extractfields cd X μy weights weightsum
    @extractfields X rowval nzval colptr
    μ = Intercept ? cd.μX[ipred] : zero(T)
    ssq = zero(T)
    ty = zero(T)

    zeroweightsum = weightsum
    zeroweighty = zero(T)
    @inbounds @simd for i = colptr[ipred]:colptr[ipred+1]-1
        row = rowval[i]
        ssq += abs2(nzval[i] - μ)*weights[row]
        ty += (y[row] - μy)*weights[row]*(nzval[i] - μ)
        zeroweightsum -= weights[row]
        zeroweighty -= (y[row] - μy)*weights[row]
    end

    # Correct for zero values
    ssq += abs2(μ)*zeroweightsum
    ty -= μ*zeroweighty

    (ssq, ty)
end

# Compute:
# XtX = (X .- μX')'*(weights.*(X[:, icoef] - μX[icoef]))
#     = X'*(weights.*(X[:, icoef] - μX[icoef])) - μX*weights'*(X[:, icoef] - μX[icoef])
#     = X'*(weights.*(X[:, icoef] - μX[icoef]))
# since μX'*weights'*(X[:, icoef] - μX[icoef])
#     = \sum_{i=1}^n w*(X_{i, icoef} - \frac{\sum_{j=1}^n X_{j, icoef} w_j}{\sum_{j=1}^n w})
#     = 0
#
# This must be called for all icoef < icoef before it may be called
# for a given icoef
function computeXtX!(cd::CovarianceCoordinateDescent{T,Intercept},
                      coef::SparseCoefficients{T}, icoef::Int, ipred::Int) where {T,Intercept}
    @extractfields cd X tmp weights XtX μX

    @simd for i = 1:size(X, 1)
        @inbounds tmp[i] = X[i, ipred]*weights[i]
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
                μ = Intercept ? μX[jpred] : zero(T)
                @simd for i = 1:size(X, 1)
                    s += tmp[i]*(X[i, jpred] - μ)
                end
                XtX[icoef, jpred] = s
            end
        end
    end
    cd
end

# In the sparse case, we actually compute:
# XtX = X'*(weights.*X[:, icoef]) - μX*μX[icoef]*sum(weights)
#
# TODO: Can this be done efficiently without making X[:, icoef] dense?
# TODO: Can this be done efficiently with a sparse XtX, e.g by
#       saving only the first term above? Storage may be a problem.
function computeXtX!(cd::CovarianceCoordinateDescent{T,Intercept,M},
                     coef::SparseCoefficients{T}, icoef::Int, ipred::Int) where {T,Intercept,M<:SparseMatrixCSC}
    @extractfields cd X tmp weights weightsum XtX μX
    @extractfields X colptr rowval nzval

    fill!(tmp, zero(T))
    μi = Intercept ? μX[ipred] : zero(T)
    @inbounds @simd for i = colptr[ipred]:colptr[ipred+1]-1
        tmp[rowval[i]] = nzval[i]*weights[rowval[i]]
    end

    @inbounds for jpred = 1:size(X, 2)
        if jpred == ipred
            XtX[icoef, jpred] = cd.Xssq[jpred] # + abs2(μi)*weightsum
        else
            jcoef = coef.predictor2coef[jpred]
            if 0 < jcoef < icoef
                XtX[icoef, jpred] = XtX[jcoef, ipred]
            else
                s = zero(T)
                @simd for i = colptr[jpred]:colptr[jpred+1]-1
                    @inbounds s += tmp[rowval[i]]*nzval[i]
                end
                μj = Intercept ? μX[jpred] : zero(T)
                XtX[icoef, jpred] = s - μi*μj*weightsum
            end
        end
    end
    cd
end

# Updates CoordinateDescent object with (possibly) new y vector and
# weights
function update!(cd::CovarianceCoordinateDescent{T,Intercept,M},
                 coef::SparseCoefficients{T}, y::Vector{T}, wt::Vector{T}) where {T,Intercept,M}
    @extractfields cd X Xty μX Xssq weights

    copyto!(weights, wt)
    weightsum = cd.weightsum = sum(weights)
    weightsuminv = inv(weightsum)

    # Compute μy
    μy = zero(T)
    if Intercept
        @simd for i = 1:length(y)
            @inbounds μy += y[i]*weights[i]
        end
        μy *= weightsuminv
        cd.μy = μy
    end

    cd.yty = computeyty(cd, y)

    if Intercept
        # Compute weighted mean
        computeμX!(cd)
    end

    # Compute Xssq and X'y
    for j = 1:size(X, 2)
        Xssq[j], Xty[j] = computeXssqXty(cd, y, j)
    end

    # Compute X'X for predictors in model
    for icoef = 1:nnz(coef)
        computeXtX!(cd, coef, icoef, coef.coef2predictor[icoef])
    end

    cd
end

# This is abstracted out in case someday we want to make XtX sparse
getXtX(::CovarianceCoordinateDescent, XtX, jcoef, ipred) = XtX[jcoef, ipred]

function compute_gradient(cd::CovarianceCoordinateDescent{T}, XtX, coef, ipred) where T
    s = zero(T)
    @simd for jcoef = 1:nnz(coef)
        @inbounds s += getXtX(cd, XtX, jcoef, ipred)*coef.coef[jcoef]
    end
    s
end

# Performs the cycle of all predictors
function cycle!(coef::SparseCoefficients{T}, cd::CovarianceCoordinateDescent{T}, λ::T, all::Bool) where T
    @extractfields cd X Xty XtX Xssq α ω

    maxdelta = zero(T)
    if all
        @inbounds for ipred = 1:length(Xty)
            # Use all predictors for first and last iterations
            s = Xty[ipred] - compute_gradient(cd, XtX, coef, ipred)

            icoef = coef.predictor2coef[ipred]
            if icoef != 0
                oldcoef = coef.coef[icoef]
                s += getXtX(cd, XtX, icoef, ipred)*oldcoef
            else
                oldcoef = zero(T)
            end

            λωj = λω(λ,ω,ipred)
            _ssq = Xssq[ipred]
            newcoef = _ssq > zero(T) ? S(s, λωj*α)/(_ssq + λωj*(1 - α)) : zero(T)
            if oldcoef != newcoef
                if icoef == 0
                    # Adding a new variable to the model
                    nnz(coef) > cd.maxncoef &&
                        error("maximum number of coefficients $(cd.maxncoef) exceeded at λ = $λ (λωj=$λωj)")
                    icoef = addcoef!(coef, ipred)
                    cd.coefitr = addcoef(cd.coefitr, icoef)

                    # Compute cross-product with predictors
                    computeXtX!(cd, coef, icoef, ipred)
                end
                maxdelta = max(maxdelta, abs2(oldcoef - newcoef)*Xssq[ipred])
                coef.coef[icoef] = newcoef
            end
        end
    else
        # Iterate over only the predictors already in the model
        @inbounds for icoef = cd.coefitr
            ipred = coef.coef2predictor[icoef]
            oldcoef = coef.coef[icoef]
            oldcoef == 0 && continue
            s = Xty[ipred] + getXtX(cd, XtX, icoef, ipred)*oldcoef - compute_gradient(cd, XtX, coef, ipred)
            λωj = λω(λ,ω,ipred)
            newcoef = coef.coef[icoef] = S(s, λωj*α)/(Xssq[ipred] + λωj*(1 - α))
            maxdelta = max(maxdelta, abs2(oldcoef - newcoef)*Xssq[ipred])
        end
    end
    maxdelta
end

# y'y - 2β'X'y + β'X'Xβ
function ssr(coef::SparseCoefficients{T}, cd::CovarianceCoordinateDescent{T}) where T
    XtX = cd.XtX
    Xty = cd.Xty
    v = cd.yty
    @inbounds for icoef = 1:nnz(coef)
        ipred = coef.coef2predictor[icoef]
        s = -2*Xty[ipred] + compute_gradient(cd, XtX, coef, ipred)
        v += coef.coef[icoef]*s
    end
    v
end

# Value of the intercept
intercept(coef::SparseCoefficients{T}, cd::CovarianceCoordinateDescent{T,true}) where {T} =
    cd.μy - dot(cd.μX, coef)

# Value of the linear predictor
function linpred!(mu::Vector{T}, cd::CovarianceCoordinateDescent{T},
                  coef::SparseCoefficients{T}, b0::T) where T
    mul!(mu, cd.X, coef)
    if b0 != 0
        @simd for i = 1:length(mu)
            @inbounds mu[i] += b0
        end
    end
    mu
end

function cdfit!(coef::SparseCoefficients{T}, cd::CoordinateDescent{T}, λ, criterion) where T
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
    for outer iter = 1:maxiter
        oldb0 = b0
        maxdelta = cycle!(coef, cd, λ, converged)
        b0 = intercept(coef, cd)
        maxdelta = max(maxdelta, abs2(oldb0 - b0)*cd.weightsum)

        # Test for convergence
        prev_converged = converged
        if criterion == :obj
            objold = obj
            dev = ssr(coef, cd)
            obj = dev/2 + λ*P(cd.α, coef, cd.ω)
            converged = abs(objold - obj) < tol*obj
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

# Compute working response based on eta and working residuals
function wrkresp!(out, eta, wrkresid, offset)
    if isempty(offset)
        @simd for i = 1:length(out)
            @inbounds out[i] = eta[i] + wrkresid[i]
        end
    else
        @simd for i = 1:length(out)
            @inbounds out[i] = eta[i] + wrkresid[i] - offset[i]
        end
    end
    out
end

poststep(path::LassoPath, cd::CoordinateDescent, i::Int, coefs::SparseCoefficients) = nothing

# Fits GLMs (outer and middle loops)
function StatsBase.fit!(path::RegularizationPath{S,T}; verbose::Bool=false, irls_maxiter::Int=30,
                        cd_maxiter::Int=100000, cd_tol::Real=1e-7, irls_tol::Real=1e-7,
                        stopearly::Bool=true, # whether to break path when little change in dev and trim the λ set
                        criterion=:coef, minStepFac::Real=0.001) where {S<:GeneralizedLinearModel,T}
    irls_maxiter >= 1 || error("irls_maxiter must be positive")
    0 < minStepFac < 1 || error("minStepFac must be in (0, 1)")
    criterion == :obj || criterion == :coef || error("criterion must be obj or coef")

    @extractfields path nulldev λ autoλ Xnorm m
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

    @extractfields cd X α ω
    @extractfields r offset eta wrkresid wrkwt
    coefs = spzeros(T, size(X, 2), nλ)
    b0s = zeros(T, nλ)
    oldcoef = SparseCoefficients{T}(size(X, 2))
    newcoef = SparseCoefficients{T}(size(X, 2))
    pct_dev = zeros(nλ)
    dev_ratio = convert(T, NaN)
    dev = convert(T, NaN)
    b0 = zero(T)
    scratchmu = Vector{T}(undef, size(X, 1))
    objold = convert(T, Inf)

    if autoλ && isa(ω, Nothing)
        # No need to fit the first model
        b0s[1] = path.nullb0
        i = 2
    else
        i = 1
    end

    dev = NaN
    niter = 0
    if nλ == 0
        i = 0
    elseif i <= nλ  # need this check because it is possible that autoλ is true and nλ is 1
        while true # outer loop
            obj = convert(T, Inf)
            last_dev_ratio = dev_ratio
            curλ = λ[i]

            converged = false

            for iirls=1:irls_maxiter # middle loop
                copyto!(oldcoef, newcoef)
                oldb0 = b0

                # Compute working response
                wrkresp!(scratchmu, eta, wrkresid, offset)

                # Run coordinate descent inner loop
                niter += cdfit!(newcoef, update!(cd, newcoef, scratchmu, wrkwt), curλ, criterion)
                b0 = intercept(newcoef, cd)

                # Update GLM and get deviance
                updateμ!(r, linpred!(scratchmu, cd, newcoef, b0))

                # Compute Elastic Net objective
                objold = obj
                dev = deviance(r)
                obj = dev/2 + curλ*P(α, newcoef, ω)

                if obj > objold + length(scratchmu)*eps(objold)
                    verbose && println("step-halving because obj=$obj > $objold + $(length(scratchmu)*eps(objold)) = length(scratchmu)*eps(objold)")
                    f = 1.0
                    b0diff = b0 - oldb0
                    coefdiff = SparseCoefficients{T}(size(X, 2))
                    copyto!(coefdiff,newcoef)
                    for icoef = 1:nnz(oldcoef)
                        coefdiff.coef[icoef] -= oldcoef.coef[icoef]
                    end
                    while obj > objold
                        # verbose && println("f=$f: $obj > $objold, dev=$dev, b0=$b0, newcoef=$newcoef")
                        f /= 2.; f > minStepFac || error("step-halving failed at beta = $(newcoef)")
                        for icoef = 1:nnz(newcoef)
                            oldcoefval = icoef > nnz(oldcoef) ? zero(T) : oldcoef.coef[icoef]
                            newcoef.coef[icoef] = oldcoefval+f*(coefdiff.coef[icoef])
                        end
                        b0 = oldb0+f*b0diff
                        updateμ!(r, linpred!(scratchmu, cd, newcoef, b0))
                        dev = deviance(r)
                        obj = dev/2 + curλ*P(α, newcoef, ω)
                    end
                end

                # Determine if we have converged
                if criterion == :obj
                    converged = abs(objold - obj) < irls_tol*obj
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
            if i == nλ || (stopearly && autoλ && (last_dev_ratio - dev_ratio < MIN_DEV_FRAC_DIFF ||
                           pct_dev[i] > MAX_DEV_FRAC))
                break
            end

            verbose && println("$i: λ=$curλ, pct_dev=$(pct_dev[i])")
            poststep(path, cd, i, newcoef)
            i += 1
        end
    end

    i = min(i, nλ)
    path.λ = path.λ[1:i]
    path.pct_dev = pct_dev[1:i]
    path.coefs = coefs[:, 1:i]
    path.b0 = b0s[1:i]
    path.niter = niter
    if !isempty(Xnorm)
        lmul!(Diagonal(Xnorm), path.coefs)
    end
end

# Fits linear models (just the outer loop)
function StatsBase.fit!(path::RegularizationPath{S,T}; verbose::Bool=false,
                        cd_maxiter::Int=10000, cd_tol::Real=1e-7, irls_tol::Real=1e-7,
                        stopearly::Bool=true, # whether to break path when little change in dev and trim the λ set
                        criterion=:coef, minStepFac::Real=eps()) where {S<:LinearModel,T}
    0 < minStepFac < 1 || error("minStepFac must be in (0, 1)")
    criterion == :obj || criterion == :coef || error("criterion must be obj or coef")

    @extractfields path nulldev λ autoλ Xnorm m
    nλ = length(λ)
    r = m.rr
    cd = m.pp
    cd.maxiter = cd_maxiter
    cd.tol = cd_tol

    @extractfields cd X ω
    coefs = spzeros(T, size(X, 2), nλ)
    b0s = zeros(T, nλ)
    newcoef = SparseCoefficients{T}(size(X, 2))
    pct_dev = zeros(nλ)
    dev_ratio = convert(T, NaN)
    niter = 0

    update!(cd, newcoef, r.mu, r.wts)

    if autoλ && isa(ω, Nothing)
        # No need to fit the first model
        b0s[1] = path.nullb0
        i = 2
    else
        i = 1
    end

    if i <= nλ  # need this check because it is possible that autoλ is true and nλ is 1
        while true # outer loop
            last_dev_ratio = dev_ratio
            curλ = λ[i]

            # Run coordinate descent
            niter += cdfit!(newcoef, cd, curλ, criterion)

            dev_ratio = cd.dev/nulldev
            pct_dev[i] = 1 - dev_ratio
            addcoefs!(coefs, newcoef, i)
            b0s[i] = intercept(newcoef, cd)

            # Test whether we should continue
            if i == nλ || (stopearly && autoλ && (last_dev_ratio - dev_ratio < MIN_DEV_FRAC_DIFF ||
                           pct_dev[i] > MAX_DEV_FRAC))
                break
            end

            verbose && println("$i: λ=$curλ, pct_dev=$(pct_dev[i])")
            poststep(path, cd, i, newcoef)
            i += 1
        end
    end

    i = min(i, nλ)
    path.λ = path.λ[1:i]
    path.pct_dev = pct_dev[1:i]
    path.coefs = coefs[:, 1:i]
    path.b0 = b0s[1:i]
    path.niter = niter
    if !isempty(Xnorm)
        lmul!(Diagonal(Xnorm), path.coefs)
    end
end
