# using Reexport, StatsBase, ..Util, Compat
# @reexport using GLM, Lasso
# using GLM.FPVector, GLM.wrkwt!
# export GammaLassoPath

# Implements the Taddy (2016)
# One-step estimator paths for concave regularization. arXiv
# Preprint 	arXiv:1308.5623v8. Retrieved from
# http://arxiv.org/abs/1308.5623

## GAMMA LASSO PATH

type GammaLassoPath{S<:@compat(Union{LinearModel,GeneralizedLinearModel}),T} <: RegularizationPath
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

function Base.show(io::IO, path::GammaLassoPath)
    coefs = path.coefs
    ncoefs = zeros(Int, size(coefs, 2))
    p = size(coefs,1)

    prefix = isa(path.m, GeneralizedLinearModel) ? string(typeof(distfun(path)).name.name, " ") : ""
    if path.γ!=zeros(p)
        prefix*="Gamma-"
    end
    println(io, prefix*"Lasso Solution Path ($(size(path.coefs, 2)) solutions for $(size(path.coefs, 1)) predictors in $(path.niter) iterations):")

    for i = 1:size(coefs, 2)-1
        ncoefs[i] = coefs.colptr[i+1] - coefs.colptr[i]
    end
    ncoefs[end] = nnz(coefs) - coefs.colptr[size(coefs, 2)] + 1
    Base.showarray(io, [path.λ path.pct_dev df(path) aicc(path)]; header=false)
end

# ## Error handling, holds a copy of the current path to debug
# immutable ConvergenceException <: Exception
#     msg
#     debugvars
# end
#

## MODEL CONSTRUCTION

"Compute coeffiecient specific weights vector ω_j^t based on previous iteration coefficients β"
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

function StatsBase.fit{T<:AbstractFloat,V<:FPVector}(::Type{GammaLassoPath},
                                                     X::AbstractMatrix{T}, y::V, d::UnivariateDistribution=Normal(),
                                                     l::Link=canonicallink(d);
                                                     γ::@compat(Union{Number,Vector{Number}})=0.0,
                                                     wts::@compat(Union{FPVector,Void})=ones(T, length(y)),
                                                     offset::AbstractVector=similar(y, 0),
                                                     α::Number=one(eltype(y)), nλ::Int=100,
                                                     λminratio::Number=ifelse(size(X, 1) < size(X, 2), 0.01, 1e-4),
                                                     λ::@compat(Union{Vector,Void})=nothing, standardize::Bool=true,
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
        X = scale(X, Xnorm)
    else
        Xnorm = T[]
    end

    # gamma lasso adaptation
    # can potentially pass a different γ for each element of X, but if scalar we copy it to all params
    p = size(X, 2)
    if length(γ) == 1
      γ=convert(Vector{T},repmat([γ],p))
    else
      @assert length(γ)==p "length(γ) != number of parameters ($p)"
    end

    # initialize penalty factors to 1 (no rescaling to sum to the number of coefficients)
    ω = ones(T,p)
    # ω = rescale(ω,p)

    # Lasso initialization
    α = convert(T, α)
    λminratio = convert(T, λminratio)
    coefitr = randomize ? RandomCoefficientIterator() : (1:0)
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

# Fits GLMs (outer and middle loops)
function StatsBase.fit!{S<:GeneralizedLinearModel,T}(path::GammaLassoPath{S,T}; verbose::Bool=false, irls_maxiter::Int=30,
                                                     cd_maxiter::Int=100000, cd_tol::Real=1e-7, irls_tol::Real=1e-7,
                                                     criterion=:coef, minStepFac::Real=0.001)
    irls_maxiter >= 1 || error("irls_maxiter must be positive")
    0 < minStepFac < 1 || error("minStepFac must be in (0, 1)")
    criterion == :obj || criterion == :coef || error("criterion must be obj or coef")

    @extractfields path nulldev λ autoλ Xnorm m γ
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
    @extractfields r offset eta wrkresid
    coefs = spzeros(T, size(X, 2), nλ)
    b0s = zeros(T, nλ)
    oldcoef = SparseCoefficients{T}(size(X, 2))
    newcoef = SparseCoefficients{T}(size(X, 2))
    pct_dev = zeros(nλ)
    dev_ratio = convert(T, NaN)
    dev = convert(T, NaN)
    b0 = zero(T)
    scratchmu = Array(T, size(X, 1))
    objold = convert(T, Inf)

    if autoλ
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
    else
        while true # outer loop
            obj = convert(T, Inf)
            last_dev_ratio = dev_ratio
            curλ = λ[i]

            converged = false

            for iirls=1:irls_maxiter # middle loop
                copy!(oldcoef, newcoef)
                oldb0 = b0

                # Compute working response
                wrkresp!(scratchmu, eta, wrkresid, offset)
                wrkwt = wrkwt!(r)

                # Run coordinate descent inner loop
                niter += cdfit!(newcoef, update!(cd, newcoef, scratchmu, wrkwt), curλ, criterion)
                b0 = intercept(newcoef, cd)

                # Update GLM and get deviance
                updatemu!(r, linpred!(scratchmu, cd, newcoef, b0))

                # Compute Elastic Net objective
                objold = obj
                dev = deviance(r)
                obj = dev/2 + curλ*P(α, newcoef, ω)

                if obj > objold + length(scratchmu)*eps(objold)
                    verbose && println("step-halving because obj=$obj > $objold + $(length(scratchmu)*eps(objold)) = length(scratchmu)*eps(objold)")
                    f = 1.0
                    b0diff = b0 - oldb0
                    coefdiff = SparseCoefficients{T}(size(X, 2))
                    copy!(coefdiff,newcoef)
                    for icoef = 1:nnz(newcoef)
                        oldcoefval = icoef > nnz(oldcoef) ? zero(T) : oldcoef.coef[icoef]
                        coefdiff.coef[icoef] = newcoef.coef[icoef] - oldcoefval
                    end
                    while obj > objold
                        verbose && println("f=$f: $obj > $objold, dev=$dev, b0=$b0, newcoef=$newcoef")
                        # f /= 2.; f > minStepFac || throw(ConvergenceException("step-halving failed at beta = $(newcoef)",(obj,objold,path,f,newcoef,oldcoef,b0,oldb0,b0diff,coefdiff,scratchmu,cd,r,α,curλ)))
                        f /= 2.; f > minStepFac || error("step-halving failed at beta = $(newcoef)")
                        for icoef = 1:nnz(newcoef)
                            oldcoefval = icoef > nnz(oldcoef) ? zero(T) : oldcoef.coef[icoef]
                            newcoef.coef[icoef] = oldcoefval+f*(coefdiff.coef[icoef])
                        end
                        b0 = oldb0+f*b0diff
                        updatemu!(r, linpred!(scratchmu, cd, newcoef, b0))
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
            if i == nλ || (autoλ && (last_dev_ratio - dev_ratio < MIN_DEV_FRAC_DIFF ||
                           pct_dev[i] > MAX_DEV_FRAC))
                break
            end

            verbose && println("$i: λ=$curλ, pct_dev=$(pct_dev[i])")
            i += 1
            computeω!(ω,γ,newcoef) # use β^{i-1} for β^{i} gamma lasso weights
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
function StatsBase.fit!{S<:LinearModel,T}(path::GammaLassoPath{S,T}; verbose::Bool=false,
                                          cd_maxiter::Int=10000, cd_tol::Real=1e-7, irls_tol::Real=1e-7,
                                          criterion=:coef, minStepFac::Real=eps())
    0 < minStepFac < 1 || error("minStepFac must be in (0, 1)")
    criterion == :obj || criterion == :coef || error("criterion must be obj or coef")

    @extractfields path nulldev λ autoλ Xnorm m γ
    nλ = length(λ)
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
        niter += cdfit!(newcoef, cd, curλ, criterion)

        dev_ratio = cd.dev/nulldev
        pct_dev[i] = 1 - dev_ratio
        addcoefs!(coefs, newcoef, i)
        b0s[i] = intercept(newcoef, cd)

        # Test whether we should continue
        if i == nλ || (autoλ && last_dev_ratio - dev_ratio < MIN_DEV_FRAC_DIFF ||
                       pct_dev[i] > MAX_DEV_FRAC)
            break
        end

        verbose && println("$i: λ=$curλ, pct_dev=$(pct_dev[i])")
        i += 1
        computeω!(cd.ω,γ,newcoef) # use β^{i-1} for β^{i} gamma lasso weights
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
