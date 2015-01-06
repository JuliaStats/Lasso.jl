module Lasso
import Base.LinAlg.BlasReal

using Reexport, StatsBase
@reexport using GLM, Distributions
using GLM.FPVector, GLM.wrkwt!
export LassoPath, fit

## COORDINATE DESCENT ROUTINES
S(z, γ) = ifelse(γ >= abs(z), zero(z), ifelse(z > 0, z - γ, z + γ))

function P{T}(α::T, β::Vector{T})
    x = zero(T)
    @inbounds @simd for i = 1:length(β)
        x += (1 - α)/2*abs2(β[i]) + α*abs(β[i])
    end
    x
end

abstract CoordinateDescent{T} <: LinPred
type NaiveCoordinateDescent{T} <: CoordinateDescent{T}
    X::Matrix{T}                  # original design matrix
    μy::T                         # mean of y at current weights
    μX::Vector{T}                 # mean of X at current weights
    Xssq::Vector{T}               # weighted sum of squares of each column of X
    residuals::Vector{T}          # y - Xβ (unscaled with centered X)
    weights::Vector{T}            # weights for each observation
    dev::T                        # last deviance
    intercept::Bool               # whether an intercept should be fitted
    α::T                          # elastic net parameter
    maxiter::Int                  # maximum number of iterations
    tol::T                        # tolerance as ratio of deviance to null deviance
    abstol::T                     # tolerance in units of deviance

    NaiveCoordinateDescent{T}(X::Matrix{T}, intercept::Bool, α::T, maxiter::Int, tol::T)  =
        new(X, zero(T), zeros(T, size(X, 2)), ones(T, size(X, 2)),
            fill(convert(T, NaN), size(X, 1)), fill(convert(T, NaN), size(X, 1)),
            convert(T, NaN), intercept, α, maxiter, tol, convert(T, NaN))
end

# Updates CoordinateDescent object with (possibly) new y vector and
# weights
function update!{T}(cd::NaiveCoordinateDescent{T}, coef::Vector{T}, y::Vector{T}, wt::Vector{T}, scratch::Any)
    residuals = cd.residuals
    X = cd.X
    weights = copy!(cd.weights, wt)
    # weights = scale!(cd.weights, wt, 1/length(y))
    weightsuminv = inv(sum(weights))

    A_mul_B!(residuals, X, coef)
    @simd for i = 1:length(y)
        @inbounds residuals[i] = y[i] - residuals[i]
    end

    if cd.intercept
        μX = cd.μX
        Xssq = cd.Xssq
        @inbounds for j = 1:size(X, 2)
            # Update μX
            μ = zero(T)
            @simd for i = 1:size(X, 1)
                μ += X[i, j]*weights[i]
            end
            μX[j] = μ*weightsuminv

            # Update Xssq
            ws = zero(T)
            @simd for i = 1:size(X, 1)
                ws += abs2(X[i, j] - μ)*weights[i]
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

        # Update abstol
        nulldev = zero(T)
        @simd for i = 1:length(y)
            @inbounds nulldev += abs2(y[i] - μy)*weights[i]
        end
        cd.abstol = nulldev * cd.tol

        # Center residuals
        @simd for i = 1:length(residuals)
            @inbounds residuals[i] -= μres
        end
    else
        Xssq = cd.Xssq
        @inbounds for j = 1:size(X, 2)
            ws = zero(T)
            @simd for i = 1:size(X, 1)
                ws += abs2(X[i, j])*weights[i]
            end
            Xssq[j] = ws
        end

        nulldev = zero(T)
        @simd for i = 1:length(y)
            nulldev += abs2(y[i])*weights[i]
        end
        cd.abstol = nulldev * cd.tol
    end

    cd
end

# Performs the cycle of all predictors
function cycle!{T}(coef::Vector{T}, cd::NaiveCoordinateDescent{T}, λ::T, α::T, all::Bool)
    X = cd.X
    residuals = cd.residuals
    weights = cd.weights
    Xssq = cd.Xssq
    μX = cd.μX

    offset = 1
    @inbounds for j = 1:size(X, 2)
        # Use all variables for first and last iterations
        if all || coef[j] != 0
            oldcoef = coef[j]

            # Update coefficient
            v = Xssq[j]*coef[j]
            @simd for i = 1:size(X, 1)
                v += (X[i, j] - μX[j])*residuals[i]*weights[i]
            end
            coef[j] = S(v, λ*α)/(Xssq[j] + λ*(1 - α))
            # println("s $j => $v, den = $((Xssq[j] + λ*(1 - α)))")
            # println("$j => $(coef[j])")

            # Update residual
            BLAS.axpy!(size(X, 1), oldcoef - coef[j], pointer(X, offset), 1, residuals, 1)
        end
        offset += size(X, 1)
    end
end

# Sum of squared residuals. Residuals are always up to date
function ssr{T}(coef::Vector{T}, cd::NaiveCoordinateDescent{T})
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
    XtX::Matrix{T}                # X'X (scaled by weights)
    activeset::Vector{Int}        # set of non-zero predictors
    scratch::Vector{T}            # scratch for residual calculation
    dev::T                        # last deviance
    intercept::Bool               # whether an intercept should be fitted
    α::T                          # elastic net parameter
    maxiter::Int                  # maximum number of iterations
    tol::T                        # tolerance as ratio of deviance to null deviance
    abstol::T                     # tolerance in units of deviance

    function CovarianceCoordinateDescent{T}(X::Matrix{T}, intercept::Bool, α::T, maxiter::Int, tol::T)
        activeset = Int[]
        sizehint!(activeset, max(size(X, 1), size(X, 2)))
        new(X, zero(T), zeros(T, size(X, 2)), convert(T, NaN), Array(T, size(X, 2)),
            Array(T, size(X, 2), size(X, 2)), activeset, Array(T, size(X, 2)), convert(T, NaN),
            intercept, α, maxiter, tol, convert(T, NaN))
    end
end

# Updates CoordinateDescent object with (possibly) new y vector and
# weights
function update!{T}(cd::CovarianceCoordinateDescent{T}, ::Vector{T}, y::Vector{T}, wt::Vector{T}, scratch::Matrix{T})
    X = cd.X
    Xty = cd.Xty
    μX = cd.μX
    wtsuminv = inv(sum(wt))
    if cd.intercept
        # Compute μy
        μy = zero(T)
        @simd for i = 1:length(y)
            @inbounds μy += y[i]*wt[i]
        end
        μy *= wtsuminv
        cd.μy = μy
        # @test_approx_eq_eps μy mean(y, weights(wt)) 50*eps()

        # Compute y'y
        yty = zero(T)
        @simd for i = 1:length(y)
            @inbounds yty += abs2(y[i] - μy)*wt[i]
        end
        # ymμ = y .- μy
        # ymμ .*= sqrt(wt)
        # @test_approx_eq yty dot(ymμ, ymμ)

        # TODO maybe don't do this for all columns until they are in the model
        for j = 1:size(scratch, 2)
            μ = zero(T)
            # Compute weighted mean
            @inbounds @simd for i = 1:size(scratch, 1)
                μ += X[i, j]*wt[i]
            end
            μ *= wtsuminv
            μX[j] = μ
            # @test_approx_eq_eps μ mean(X[:, j], weights(wt)) sqrt(eps())

            # Subtract weighted mean
            @simd for i = 1:size(scratch, 1)
                @inbounds scratch[i, j] = (X[i, j] - μ)*wt[i]
            end

            # Compute X'y
            v = zero(T)
            @simd for i = 1:size(scratch, 1)
                @inbounds v += (y[i] - μy)*scratch[i, j]
            end
            Xty[j] = v
            # @test_approx_eq Xty[j] Xmμ[:, j]'*ymμ
        end
    else
        # Xmμ = X.*sqrt(wt)
        yty = zero(T)
        @simd for i = 1:length(y)
            @inbounds yty += abs2(y[i])*wt[i]
        end
        broadcast!(*, scratch, cd.X, wt)
        Ac_mul_B!(Xty, scratch, y)
    end
    cd.yty = yty
    cd.abstol = yty * cd.tol
    Ac_mul_B!(cd.XtX, X, scratch)
    # @test_approx_eq cd.XtX Xmμ'Xmμ

    cd
end

# Performs the cycle of all predictors
function cycle!{T}(coef::Vector{T}, cd::CovarianceCoordinateDescent{T}, λ::T, α::T, all::Bool)
    Xty = cd.Xty
    XtX = cd.XtX
    activeset = cd.activeset

    offset = 1
    if all
        @inbounds for j = 1:size(XtX, 1)
            # Use all variables for first and last iterations
            s = Xty[j]
            for i in activeset
                i == j && continue
                s -= XtX[i, j]*coef[i]
            end
            oldcoef = coef[j]
            newcoef = coef[j] = S(s, λ*α)/(XtX[j, j] + λ*(1 - α))
            if oldcoef == 0 && newcoef != 0
                push!(activeset, j)
            elseif oldcoef != 0 && newcoef == 0
                deleteat!(activeset, findfirst(activeset, j))
            end
            # println("s $j => $s, den = $((XtX[j, j] + λ*(1 - α)))")
            # println("$j => $(coef[j])")
            offset += size(XtX, 1)
        end
    else
        @inbounds for j in activeset
            s = Xty[j]
            for i in activeset
                i == j && continue
                s -= XtX[i, j]*coef[i]
            end
            oldcoef = coef[j]
            newcoef = coef[j] = S(s, λ*α)/(XtX[j, j] + λ*(1 - α))
            if oldcoef != 0 && newcoef == 0
                deleteat!(activeset, findfirst(activeset, j))
            end
        end
    end
    # println()
end

# y'y - 2β'X'y + β'X'Xβ
function ssr{T}(coef::Vector{T}, cd::CovarianceCoordinateDescent{T})
    # v = cd.yty - 2*dot(coef, cd.Xty) + dot(coef, BLAS.symv!('U', one(T), cd.XtX, coef, zero(T), cd.scratch))
    XtX = cd.XtX
    Xty = cd.Xty
    v = cd.yty
    activeset = cd.activeset
    @inbounds for j in activeset
        s = -2*Xty[j]
        for i in activeset
            s += XtX[i, j]*coef[i]
        end
        v += coef[j]*s
    end
    v
end

function fit!{T}(coef::Vector{T}, cd::CoordinateDescent{T}, λ)
    α = cd.α
    maxiter = cd.maxiter
    abstol = cd.abstol
    n = size(cd.X, 1)

    obj = convert(T, Inf)
    dev = convert(T, Inf)
    prev_converged = false
    converged = true

    for iter = 1:maxiter
        cycle!(coef, cd, λ, α, converged)

        # Test for convergence
        dev = ssr(coef, cd)
        newobj = dev/2 + λ*P(α, coef)
        prev_converged = converged
        converged = abs(newobj - obj) < abstol
        obj = newobj

        # Require two converging steps to return. After the first, we
        # will iterate through all variables
        prev_converged && converged && break
    end

    (!prev_converged || !converged) && error("coordinate descent failed to converge in $maxiter iterations at λ = $λ")

    cd.dev = dev
    coef
end

intercept{T}(coef::Vector{T}, cd::CoordinateDescent{T}) = cd.intercept ? cd.μy .- dot(cd.μX, coef) : zero(T)

function linpred!{T}(mu::Vector{T}, X::Matrix{T}, coef::Vector{T}, b0::T)
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
                                                    cd_maxiter::Int=10000, cd_tol::Real=1e-7, irls_tol::Real=1e-7,
                                                    minStepFac::Real=eps())
    irls_maxiter >= 1 || error("irls_maxiter must be positive")
    0 < minStepFac < 1 || error("minStepFac must be in (0, 1)")

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
    oldcoef = Array(T, size(X, 2))
    newcoef = zeros(T, size(X, 2))
    coefdiff = Array(T, size(X, 2))
    pct_dev = zeros(nλ)
    dev_ratio = convert(T, NaN)
    dev = convert(T, NaN)
    b0 = zero(T)
    scratch = Array(T, size(X))
    scratchmu = Array(T, size(X, 1))
    eta = r.eta
    wrkresid = r.wrkresid

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
        objold = Inf
        last_dev_ratio = dev_ratio
        curλ = λ[i]

        cvg = false

        for iirls=1:irls_maxiter # middle loop
            copy!(oldcoef, newcoef)
            oldb0 = b0

            # Compute working response
            for j = 1:length(wrkresid)
                scratchmu[j] = eta[j] + wrkresid[j]
            end
            wrkwt = wrkwt!(r)

            # Run coordinate descent inner loop
            fit!(newcoef, update!(cd, newcoef, scratchmu, wrkwt, scratch), curλ)
            b0 = intercept(newcoef, cd)

            # Update GLM and get deviance
            dev = updatemu!(r, linpred!(scratchmu, X, newcoef, b0))
            obj = dev/2 + curλ*P(α, newcoef)

            if obj > objold*(1+irls_tol)
                f = 1.0
                broadcast!(-, coefdiff, newcoef, oldcoef)
                b0diff = b0 - oldb0
                while obj > objold
                    f /= 2.; f > minStepFac || error("step-halving failed at beta = $(newcoef)")
                    for icoef = 1:length(newcoef)
                        newcoef[icoef] = oldcoef[icoef]+f*coefdiff[icoef]
                    end
                    b0 = oldb0+f*b0diff
                    dev = updatemu!(r, linpred!(scratchmu, X, newcoef, b0))
                    obj = dev/2 + curλ*P(α, newcoef)
                end
            end

            crit = (objold - obj)/obj
            if abs(crit) < irls_tol
                cvg = true
                break
            end
            objold = obj
        end
        cvg || error("IRLS failed to converge in $irls_maxiter iterations at λ = $(curλ)")

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

# Fits linear models (just an outer loop)
function StatsBase.fit{S<:LinearModel,T}(path::LassoPath{S,T}; verbose::Bool=false, irls_maxiter::Int=30,
                                         cd_maxiter::Int=10000, cd_tol::Real=1e-7, irls_tol::Real=1e-7,
                                         minStepFac::Real=eps())
    irls_maxiter >= 1 || error("irls_maxiter must be positive")
    0 < minStepFac < 1 || error("minStepFac must be in (0, 1)")

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
    newcoef = zeros(T, size(X, 2))
    pct_dev = zeros(nλ)
    dev_ratio = convert(T, NaN)

    update!(cd, newcoef, r.y, r.wts, Array(T, size(X)))

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
        fit!(newcoef, cd, curλ)

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
