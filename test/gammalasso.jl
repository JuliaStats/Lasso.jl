# Comparing with Matt Taddy's gamlr.R
# To rebuild the test cases source(gammalasso.R)
using Lasso
using CSV, GLM, DataFrames, Random, SparseArrays, LinearAlgebra

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

penaltyfactors = readcsv(joinpath(datapath,"penaltyfactors.csv"))

rtol=1e-2
Random.seed!(243214)
@testset "GammaLassoPath" begin
    @testset "$family" for (family, dist, link) in (("gaussian", Normal(), IdentityLink()), ("binomial", Binomial(), LogitLink()), ("poisson", Poisson(), LogLink()))
        data = readcsv(joinpath(datapath,"gamlr.$family.data.csv"), header=false)
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
        @testset "penalty_factor=$pf" for pf = 1:1 #size(penaltyfactors,2)
            penalty_factor = penaltyfactors[:,pf]
            if penalty_factor == ones(p)
                penalty_factor = nothing
            end
            @testset "γ=$γ" for γ in [0 2 10]
                fitname = "gamma$γ.pf$pf"
                # get gamlr.R params and estimates
                params = CSV.read(joinpath(datapath,"gamlr.$family.$fitname.params.csv"))
                fittable = CSV.read(joinpath(datapath,"gamlr.$family.$fitname.fit.csv"))
                gcoefs = convert(Matrix{Float64},readcsv(joinpath(datapath,"gamlr.$family.$fitname.coefs.csv")))
                family = params[1,Symbol("fit.family")]
                γ = params[1,Symbol("fit.gamma")]
                λ = nothing #convert(Vector{Float64},fittable[Symbol("fit.lambda")]) # should be set to nothing evenatually

                # fit julia version
                glp = fit(GammaLassoPath, X, y, dist, link; γ=γ, λminratio=0.001, penalty_factor=penalty_factor, λ=λ)

                # compare
                @test issimilarhead(glp.λ,fittable[Symbol("fit.lambda")];rtol=rtol)
                @test issimilarhead(glp.b0,fittable[Symbol("fit.alpha")];rtol=rtol)
                @test issimilarhead(full(glp.coefs'),gcoefs';rtol=rtol)
                # we follow GLM.jl convention where deviance is scaled by nobs, while in gamlr it is not
                @test issimilarhead(deviance(glp),fittable[Symbol("fit.deviance")]/nobs(glp);rtol=rtol)
                @test issimilarhead(deviance(glp,X,y),fittable[Symbol("fit.deviance")]/nobs(glp);rtol=rtol)
                # @test issimilarhead(round(df(glp)[2:end]),round(fittable[2:end,Symbol("fit.df")]))
                @test issimilarhead(loglikelihood(glp),fittable[Symbol("fit.logLik")];rtol=rtol)
                @test issimilarhead(aicc(glp),fittable[Symbol("fit.AICc")];rtol=rtol)

                # TODO: figure out why these are so off, maybe because most are corner solutions
                # and stopping rules for lambda are different
                # # what we really need all these stats for is that the AICc identifies the same minima:
                # if indmin(aicc(glp)) != endof(aicc(glp)) && indmin(fittable[Symbol("fit.AICc")]) != endof(fittable[Symbol("fit.AICc")])
                #     # interior minima
                #     println("comparing intereior AICc")
                #     @test indmin(aicc(glp)) == indmin(fittable[Symbol("fit.AICc")])
                # end

                # comparse CV, NOTE: this involves a random choice of train subsamples
                gcoefs_CVmin = vec(convert(Matrix{Float64},readcsv(joinpath(datapath,"gamlr.$family.$fitname.coefs.CVmin.csv"))))
                gcoefs_CV1se = vec(convert(Matrix{Float64},readcsv(joinpath(datapath,"gamlr.$family.$fitname.coefs.CV1se.csv"))))

                glp_CVmin = coef(glp,select=:CVmin,nCVfolds=10)
                glp_CV1se = coef(glp,select=:CV1se,nCVfolds=10)

                @test glp_CVmin ≈ gcoefs_CVmin rtol=0.35
                @test glp_CV1se ≈ gcoefs_CV1se rtol=0.35

                if γ==0
                    # Compare with LassoPath
                    lp = fit(LassoPath, X, y, dist, link; λminratio=0.001, penalty_factor=penalty_factor, λ=λ)
                    @test glp.λ == lp.λ
                    @test glp.b0 == lp.b0
                    @test glp.coefs == lp.coefs
                end
            end
        end
    end
end

## the following code is useful for understanding comparison failures

# rdist(x::Number,y::Number) = abs(x-y)/max(abs(x),abs(y))
# rdist{T<:Number,S<:Number}(x::AbstractArray{T}, y::AbstractArray{S}; norm::Function=vecnorm) = norm(x - y) / max(norm(x), norm(y))
# #
# (family, dist, link) = (("gaussian", Normal(), IdentityLink()), ("binomial", Binomial(), LogitLink()), ("poisson", Poisson(), LogLink()))[1]
# data = readcsv(joinpath(datapath,"gamlr.$family.data.csv"))
# y = data[:,1]
# X = data[:,2:end]
# (n,p) = size(X)
# pf = (1:size(penaltyfactors,2))[2]
# penalty_factor = penaltyfactors[:,pf]
# if penalty_factor == ones(p)
#     penalty_factor = nothing
# end
# penalty_factor
# γ = [0 2 10][2]
# fitname = "gamma$γ.pf$pf"
# # get gamlr.R params and estimates
# params = readtable(joinpath(datapath,"gamlr.$family.$fitname.params.csv"))
# fittable = readtable(joinpath(datapath,"gamlr.$family.$fitname.fit.csv"))
# gcoefs = convert(Matrix{Float64},readcsv(joinpath(datapath,"gamlr.$family.$fitname.coefs.csv")))
# family = params[1,:fit_family]
# # λ = nothing #convert(Vector{Float64},fittable[:fit_lambda]) # should be set to nothing evenatually
# λ = convert(Vector{Float64},fittable[:fit_lambda]) # should be set to nothing evenatually
# # fit julia version
# glp = fit(GammaLassoPath, X, y, dist, link, λ=λ, γ=γ, standardize=true, λminratio=0.001, penalty_factor=penalty_factor)
#
# # compare
# @test issimilarhead(glp.λ,fittable[:fit_lambda];rtol=rtol)
# @test issimilarhead(glp.b0,fittable[:fit_alpha];rtol=rtol)
# @test issimilarhead(full(glp.coefs'),gcoefs';rtol=rtol)
# rdist(full(glp.coefs'), gcoefs')
# # we follow GLM.jl convention where deviance is scaled by nobs, while in gamlr it is not
# @test issimilarhead(deviance(glp),fittable[:fit_deviance]/nobs(glp);rtol=rtol)
# @test issimilarhead(deviance(glp,X,y),fittable[:fit_deviance]/nobs(glp);rtol=rtol)
# # @test issimilarhead(round(df(glp)[2:end]),round(fittable[2:end,:fit_df]))
# @test issimilarhead(loglikelihood(glp),fittable[:fit_logLik];rtol=rtol)
# @test issimilarhead(aicc(glp),fittable[:fit_AICc];rtol=rtol)
# # what we really need all these stats for is that the AICc identifies the same minima:
# if indmin(aicc(glp)) != endof(aicc(glp)) && indmin(fittable[:fit_AICc]) != endof(fittable[:fit_AICc])
#     # interior minima
#     @test indmin(aicc(glp)) == indmin(fittable[:fit_AICc])
# end
#
# # comparse CV
# gcoefs_CVmin = vec(convert(Matrix{Float64},readcsv(joinpath(datapath,"gamlr.$family.$fitname.coefs.CVmin.csv"))))
# gcoefs_CV1se = vec(convert(Matrix{Float64},readcsv(joinpath(datapath,"gamlr.$family.$fitname.coefs.CV1se.csv"))))
#
# glp_CVmin = coef(glp,select=:CVmin,nCVfolds=10)
# glp_CV1se = coef(glp,select=:CV1se,nCVfolds=10)
#
# @test glp_CVmin ≈ gcoefs_CVmin rtol=0.3
# @test glp_CV1se ≈ gcoefs_CV1se rtol=0.3

# newX = X
# select=:all
# offset = path.m.rr.offset
# aicc(path)
# dev0=deviance(path)
# μ = predict(path, X; offset=offset, select=select)
# dev1=deviance(path,y,μ)
# dev2=deviance(path,X,y; offset=offset, select=select)
# dev0 ≈ dev1
# dev1 == dev2
#
# lp = fit(LassoPath, X, y, dist, link; λ=λ, λminratio=0.001, penalty_factor=penalty_factor) #, λminratio=0.001)
# @test glp.λ == lp.λ
# @test glp.b0 == lp.b0
# @test glp.coefs == lp.coefs
# deviance(path)
# μ = predict(path, X; offset=offset, select=select)
# deviance(path,y,μ)
# deviance(path,X,y)
#
#
#
#
#
#
#
#
#
#
#
# #
