# Comparing with Matt Taddy's gamlr.R
# To rebuild the test cases source(gammalasso.R)
using Lasso

using CSV, GLM, DataFrames, Random, SparseArrays, LinearAlgebra
using Distributions: mean, weights

# often path length is different because of different stopping rules...
function issimilarhead(a::AbstractVector,b::AbstractVector;rtol=1e-4)
    n = min(size(a,1),size(b,1))
    isapprox(a[1:n],b[1:n];rtol=rtol) ? true : [a[1:n] b[1:n]]
end

function issimilarhead(a::AbstractMatrix,b::AbstractMatrix;rtol=1e-4)
    n = min(size(a,1),size(b,1))
    m = min(size(a,2),size(b,2))
    isapprox(a[1:n,1:m],b[1:n,1:m];rtol=rtol) ? true : [a[1:n,1:m] b[1:n,1:m]]
end

datapath = joinpath(dirname(@__FILE__), "data")

penaltyfactors = readcsvmat(joinpath(datapath,"penaltyfactors.csv"))

rtol=1e-2
Random.seed!(243214)
@testset "GammaLassoPath" begin
    @testset "$family" for (family, dist, link) in (("gaussian", Normal(), IdentityLink()), ("binomial", Binomial(), LogitLink()), ("poisson", Poisson(), LogLink()))
        data = readcsvmat(joinpath(datapath,"gamlr.$family.data.csv"))
        y = convert(Vector{Float64},data[:,1])
        X = convert(Matrix{Float64},data[:,2:end])
        (n,p) = size(X)
        # TODO: we currently do not match Taddy's gamlr with a penalty_factor!=1, likely because it standardizes them differently than glmnet.
        # compared with glmnet (and γ=0), however, we get exactly the same results. Bug / feature ?
        # relevant gamlr.c code where (penalty_factor is W)
        #   if(*standardize){
        #     for(int j=0; j<p; j++){
        #       if(fabs(H[j])<1e-10){ H[j]=0.0; W[j] = INFINITY; }
        #       else W[j] *= sqrt(H[j]/vsum);
        #     }
        #   }
        @testset "penalty_factor=$pf" for pf = 1:1#size(penaltyfactors,2)
            penalty_factor = penaltyfactors[:,pf]
            if penalty_factor == ones(p)
                penalty_factor = nothing
            end
            @testset "γ=$γ" for γ in [0 2 10]
                fitname = "gamma$γ.pf$pf"

                # get gamlr.R prms and estimates
                prms = CSV.read(joinpath(datapath,"gamlr.$family.$fitname.params.csv"))
                fittable = CSV.read(joinpath(datapath,"gamlr.$family.$fitname.fit.csv"))
                gcoefs = readcsvmat(joinpath(datapath,"gamlr.$family.$fitname.coefs.csv");types=[Float64 for i=1:100])
                family = prms[1,Symbol("fit.family")]
                γ = prms[1,Symbol("fit.gamma")]
                λ = nothing #convert(Vector{Float64},fittable[Symbol("fit.lambda")]) # should be set to nothing evenatually

                # fit julia version
                glp = fit(GammaLassoPath, X, y, dist, link; γ=γ, trimλ=false,
                    λminratio=0.001, penalty_factor=penalty_factor, λ=λ,
                    standardize=true, standardize_penalty=false)

                # compare
                @test true==issimilarhead(glp.λ,fittable[Symbol("fit.lambda")];rtol=rtol)
                @test true==issimilarhead(glp.b0,fittable[Symbol("fit.alpha")];rtol=rtol)
                @test true==issimilarhead(convert(Matrix{Float64},glp.coefs'),gcoefs';rtol=rtol)
                # we follow GLM.jl convention where deviance is scaled by nobs, while in gamlr it is not
                @test true==issimilarhead(deviance(glp),fittable[Symbol("fit.deviance")]/nobs(glp);rtol=rtol)
                @test true==issimilarhead(deviance(glp,X,y),fittable[Symbol("fit.deviance")]/nobs(glp);rtol=rtol)
                # @test true==issimilarhead(round(df(glp)[2:end]),round(fittable[2:end,Symbol("fit.df")]))
                @test true==issimilarhead(loglikelihood(glp),fittable[Symbol("fit.logLik")];rtol=rtol)
                @test true==issimilarhead(aicc(glp),fittable[Symbol("fit.AICc")];rtol=rtol)

                # TODO: figure out why these are so off, maybe because most are corner solutions
                # and stopping rules for lambda are different
                # # what we really need all these stats for is that the AICc identifies the same minima:
                # if indmin(aicc(glp)) != endof(aicc(glp)) && indmin(fittable[Symbol("fit.AICc")]) != endof(fittable[Symbol("fit.AICc")])
                #     # interior minima
                #     println("comparing intereior AICc")
                #     @test indmin(aicc(glp)) == indmin(fittable[Symbol("fit.AICc")])
                # end

                # comparse CV, NOTE: this involves a random choice of train subsamples
                gcoefs_CVmin = vec(readcsvmat(joinpath(datapath,"gamlr.$family.$fitname.coefs.CVmin.csv")))
                gcoefs_CV1se = vec(readcsvmat(joinpath(datapath,"gamlr.$family.$fitname.coefs.CV1se.csv")))

                glp_CVmin = coef(glp,select=:CVmin,nCVfolds=10)
                glp_CV1se = coef(glp,select=:CV1se,nCVfolds=10)

                @test glp_CVmin ≈ gcoefs_CVmin rtol=0.35
                @test glp_CV1se ≈ gcoefs_CV1se rtol=0.35

                if γ==0
                    # Compare with LassoPath
                    lp = fit(LassoPath, X, y, dist, link; trimλ=false,
                        λminratio=0.001, penalty_factor=penalty_factor, λ=λ,
                        standardize=true, standardize_penalty=false)
                    @test glp.λ ≈ lp.λ
                    @test glp.b0 ≈ lp.b0
                    @test glp.coefs ≈ lp.coefs
                end
            end
        end
    end
end
# ## the following code is useful for understanding comparison failures
#
rdist(x::Number,y::Number) = abs(x-y)/max(abs(x),abs(y))
rdist(x::AbstractArray{T}, y::AbstractArray{S}; norm::Function=norm) where {T<:Number,S<:Number} = norm(x - y) / max(norm(x), norm(y))

rtol=1e-2
Random.seed!(243214)
(family, dist, link) = (("gaussian", Normal(), IdentityLink()),
                        ("binomial", Binomial(), LogitLink()),
                        ("poisson", Poisson(), LogLink()))[1]
data = readcsvmat(joinpath(datapath,"gamlr.$family.data.csv"))
y = convert(Vector{Float64},data[:,1])
X = convert(Matrix{Float64},data[:,2:end])
(n,p) = size(X)
# TODO: we currently do not match Taddy's gamlr with a penalty_factor!=1, likely because it standardizes them differently than glmnet.
# compared with glmnet (and γ=0), however, we get exactly the same results. Bug / feature ?
# relevant gamlr.c code where (penalty_factor is W)
#   if(*standardize){
#     for(int j=0; j<p; j++){
#       if(fabs(H[j])<1e-10){ H[j]=0.0; W[j] = INFINITY; }
#       else W[j] *= sqrt(H[j]/vsum);
#     }
#   }
pf = 1 #size(penaltyfactors,2)
penalty_factor = penaltyfactors[:,pf]
# penalty_factor = [2.0,1.0,1.0]
if penalty_factor == ones(p)
    penalty_factor = nothing
end
γ = [0 2 10][1]
fitname = "gamma$γ.pf$pf"

# get gamlr.R prms and estimates
prms = CSV.read(joinpath(datapath,"gamlr.$family.$fitname.params.csv"))
fittable = CSV.read(joinpath(datapath,"gamlr.$family.$fitname.fit.csv"))
gcoefs = readcsvmat(joinpath(datapath,"gamlr.$family.$fitname.coefs.csv");types=[Float64 for i=1:100])
family = prms[1,Symbol("fit.family")]
γ = prms[1,Symbol("fit.gamma")]

offset = Float64[]
alpha = 1.0
irls_tol = 1e-7
dofit = true
intercept = true
T = Float64
wts = ones(n)

λ = nothing #convert(Vector{Float64},fittable[Symbol("fit.lambda")]) # should be set to nothing evenatually
# λ = convert(Vector{Float64},fittable[Symbol("fit.lambda")]) # should be set to nothing evenatually

# fit julia version
glp = fit(GammaLassoPath, X, y, dist, link; γ=γ, λminratio=0.001,
    penalty_factor=penalty_factor, λ=λ, trimλ=false,
    standardize=true, standardize_penalty=false)
lp = fit(LassoPath, X, y, dist, link; λminratio=0.001,
    penalty_factor=penalty_factor, λ=λ, trimλ=true,
    standardize=true, standardize_penalty=false)

Xnorm = vec(convert(Matrix{T},std(X; dims=1, corrected=false)))
for i = 1:length(Xnorm)
    @inbounds Xnorm[i] = 1/Xnorm[i]
end
Xstd = X .* transpose(Xnorm)

ixunpenalized = findall(iszero, penalty_factor)
nullX = [ones(T, length(y), ifelse(intercept, 1, 0)) X[:, ixunpenalized]]
nullmodel = fit(GeneralizedLinearModel, nullX, y, dist, link;
                wts=wts .* T(1/sum(wts)), offset=offset, convTol=irls_tol, dofit=dofit)
nulldev = deviance(nullmodel)
nullb0 = intercept ? coef(nullmodel)[1] : zero(T)

# Find max λ
# Xy = Xstd'*broadcast!(*, nullmodel.rr.wrkresid, nullmodel.rr.wrkresid, nullmodel.rr.wrkwt)
Xy = X'*broadcast!(*, nullmodel.rr.wrkresid, nullmodel.rr.wrkresid, nullmodel.rr.wrkwt)

Lasso.rescale(penalty_factor, length(Xy))

abs.(Xy)
λmaxcands = abs.(Xy) ./ penalty_factor
λmaxcandsgamlr = [abs(Xy[i]) / penalty_factor[i] for i=1:p]
myλmaxgamlr = maximum([a for a in λmaxcandsgamlr if !isinf(a)])
Lasso.computeλ(Xy, 0.001, 1.0, 100, Lasso.rescale(penalty_factor, length(Xy)))

penalty_factor_glmnet = penalty_factor
# First fit with GLMNet
if isa(dist, Normal)
    yp = isempty(offset) ? y : y + offset
    ypstd = std(yp, corrected=false)
    # glmnet does this on entry, which changes λ mappings, but not
    # coefficients. Should we?
    yp = yp ./ ypstd
    !isempty(offset) && (offset = offset ./ ypstd)
    y = y ./ ypstd
    g = glmnet(X, yp, dist, intercept=intercept, alpha=alpha, tol=10*eps(); lambda_min_ratio=0.001, penalty_factor=penalty_factor_glmnet)
elseif isa(dist, Binomial)
    yp = zeros(size(y, 1), 2)
    yp[:, 1] = y .== 0
    yp[:, 2] = y .== 1
    g = glmnet(X, yp, dist, intercept=intercept, alpha=alpha, tol=10*eps(),
               offsets=isempty(offset) ? zeros(length(y)) : offset; lambda_min_ratio=0.001, penalty_factor=penalty_factor_glmnet)
else
    g = glmnet(X, y, dist, intercept=intercept, alpha=alpha, tol=10*eps(),
               offsets=isempty(offset) ? zeros(length(y)) : offset; lambda_min_ratio=0.001, penalty_factor=penalty_factor_glmnet)
end

### dev start
# same lambdas?
a = issimilarhead(glp.λ,fittable[Symbol("fit.lambda")];rtol=rtol)
b = issimilarhead(g.lambda,fittable[Symbol("fit.lambda")];rtol=rtol)
c = issimilarhead(g.lambda,glp.λ;rtol=rtol)
λmaxglp = glp.λ[1]
λmaxgamlr = fittable[1,Symbol("fit.lambda")]
λmaxglmnet = g.lambda[1]
λmaxglp/λmaxgamlr
glp.b0[1]

Lasso.computeλ(λmaxgamlr, 0.001, 1.0, 100)
Lasso.computeλ(λmaxglmnet, 0.001, 1.0, 100)

# same coefs?
convert(Matrix{Float64},glp.coefs')
gcoefs'
issimilarhead(convert(Matrix{Float64},glp.coefs'),gcoefs';rtol=rtol)

# same aicc?
issimilarhead(aicc(glp),fittable[Symbol("fit.AICc")];rtol=rtol)
aicc(glp)
# ixglp = Lasso.minAICc(glp)
ixglp = Lasso.minAICc(glp)
ixgamlr = argmin(fittable[Symbol("fit.AICc")])

# same coef at minAICc?
@test convert(Matrix{Float64},glp.coefs')[ixglp,:] ≈ gcoefs'[ixglp,:] rtol=rtol
@test convert(Matrix{Float64},glp.coefs')[ixglp,:] ≈ gcoefs'[ixgamlr,:] rtol=rtol

glp.b0[1]

@test glp.b0[ixglp] ≈ fittable[Symbol("fit.alpha")][ixglp] rtol=rtol
@test glp.b0[ixglp] ≈ fittable[Symbol("fit.alpha")][ixgamlr] rtol=rtol

# same deviance?
issimilarhead(deviance(glp),fittable[Symbol("fit.deviance")]/nobs(glp);rtol=rtol)
deviance(glp)[ixglp]
(fittable[Symbol("fit.deviance")]/nobs(glp))[ixglp]

# Lasso.jl stops early because
glp.pct_dev[ixglp] > Lasso.MAX_DEV_FRAC
glp.pct_dev[ixglp-1] < Lasso.MAX_DEV_FRAC

glp.pct_dev[ixglp] > Lasso.MAX_DEV_FRAC
glp.pct_dev[ixglp-1] < Lasso.MAX_DEV_FRAC

dev_ratio = 1.0 .- glp.pct_dev
dev_ratio_diff = diff(dev_ratio)

# same as glmnet?

gbeta = convert(Matrix{Float64}, g.betas)
gbeta'[ixglp,:]
gbeta'[end,:]

Vector(glp.coefs'[ixglp,:])

### dev end

# compare
@test true==issimilarhead(glp.λ,fittable[Symbol("fit.lambda")];rtol=rtol)
@test true==issimilarhead(glp.b0,fittable[Symbol("fit.alpha")];rtol=rtol)
@test true==issimilarhead(convert(Matrix{Float64},glp.coefs'),gcoefs';rtol=rtol)
# we follow GLM.jl convention where deviance is scaled by nobs, while in gamlr it is not
@test true==issimilarhead(deviance(glp),fittable[Symbol("fit.deviance")]/nobs(glp);rtol=rtol)
@test true==issimilarhead(deviance(glp,X,y),fittable[Symbol("fit.deviance")]/nobs(glp);rtol=rtol)
# @test true==issimilarhead(round(df(glp)[2:end]),round(fittable[2:end,Symbol("fit.df")]))
@test true==issimilarhead(loglikelihood(glp),fittable[Symbol("fit.logLik")];rtol=rtol)
@test true==issimilarhead(aicc(glp),fittable[Symbol("fit.AICc")];rtol=rtol)

# TODO: figure out why these are so off, maybe because most are corner solutions
# and stopping rules for lambda are different
# # what we really need all these stats for is that the AICc identifies the same minima:
# if indmin(aicc(glp)) != endof(aicc(glp)) && indmin(fittable[Symbol("fit.AICc")]) != endof(fittable[Symbol("fit.AICc")])
#     # interior minima
#     println("comparing intereior AICc")
#     @test indmin(aicc(glp)) == indmin(fittable[Symbol("fit.AICc")])
# end

# comparse CV, NOTE: this involves a random choice of train subsamples
gamlrminAICc =
gcoefs_AICc
gcoefs_CVmin = vec(readcsvmat(joinpath(datapath,"gamlr.$family.$fitname.coefs.CVmin.csv")))
gcoefs_CV1se = vec(readcsvmat(joinpath(datapath,"gamlr.$family.$fitname.coefs.CV1se.csv")))

glp_AICc = coef(glp,select=:AICc)
glp_CVmin = coef(glp,select=:CVmin,nCVfolds=10)
glp_CV1se = coef(glp,select=:CV1se,nCVfolds=10)

@test glp_CVmin ≈ gcoefs_CVmin rtol=0.35
@test glp_CV1se ≈ gcoefs_CV1se rtol=0.35
rdist(glp_CVmin,gcoefs_CVmin)
rdist(glp_CV1se,gcoefs_CV1se)

if γ==0
    # Compare with LassoPath
    lp = fit(LassoPath, X, y, dist, link; λminratio=0.001, penalty_factor=penalty_factor, λ=λ)
    @test glp.λ == lp.λ
    @test glp.b0 == lp.b0
    @test glp.coefs == lp.coefs
end
