# Comparing with Matt Taddy's gamlr.R
# To rebuild the test cases source(gammalasso.R)
using Lasso
using GLM, FactCheck, DataFrames

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

rtol=1e-2
srand(243214)
facts("GammaLassoPath") do
    for (family, dist, link) in (("gaussian", Normal(), IdentityLink()), ("binomial", Binomial(), LogitLink()), ("poisson", Poisson(), LogLink()))
        context(family) do
            data = readcsv(joinpath(datapath,"gamlr.$family.data.csv"))
            y = convert(Vector{Float64},data[:,1])
            X = convert(Matrix{Float64},data[:,2:end])
            (n,p) = size(X)
            for γ in [0 2 10]
                fitname = "gamma$γ"
                # get gamlr.R params and estimates
                params = readtable(joinpath(datapath,"gamlr.$family.$fitname.params.csv"))
                fittable = readtable(joinpath(datapath,"gamlr.$family.$fitname.fit.csv"))
                gcoefs = convert(Matrix{Float64},readcsv(joinpath(datapath,"gamlr.$family.$fitname.coefs.csv")))
                family = params[1,:fit_family]
                γ=params[1,:fit_gamma]
                # λ = convert(Vector{Float64},fittable[:fit_lambda]) # should be set to nothing evenatually
                context("γ=$γ") do

                    # fit julia version
                    glp = fit(GammaLassoPath, X, y, dist, link; γ=γ, λminratio=0.001) #, λ=λ)

                    # compare
                    @fact issimilarhead(glp.λ,fittable[:fit_lambda];rtol=rtol) --> true
                    @fact issimilarhead(glp.b0,fittable[:fit_alpha];rtol=rtol) --> true
                    @fact issimilarhead(full(glp.coefs'),gcoefs';rtol=rtol) --> true
                    # we follow GLM.jl convention where deviance is scaled by nobs, while in gamlr it is not
                    @fact issimilarhead(deviance(glp),fittable[:fit_deviance]/nobs(glp);rtol=rtol) --> true
                    @fact issimilarhead(deviance(glp,X,y),fittable[:fit_deviance]/nobs(glp);rtol=rtol) --> true
                    # @fact issimilarhead(round(df(glp)[2:end]),round(fittable[2:end,:fit_df])) --> true
                    @fact issimilarhead(loglikelihood(glp),fittable[:fit_logLik];rtol=rtol) --> true
                    @fact issimilarhead(aicc(glp),fittable[:fit_AICc];rtol=rtol) --> true

                    # TODO: figure out why these are so off, maybe because most are corner solutions
                    # and stopping rules for lambda are different
                    # # what we really need all these stats for is that the AICc identifies the same minima:
                    # if indmin(aicc(glp)) != endof(aicc(glp)) && indmin(fittable[:fit_AICc]) != endof(fittable[:fit_AICc])
                    #     # interior minima
                    #     println("comparing intereior AICc")
                    #     @fact indmin(aicc(glp)) --> indmin(fittable[:fit_AICc])
                    # end

                    # comparse CV, NOTE: this involves a random choice of train subsamples
                    gcoefs_CVmin = vec(convert(Matrix{Float64},readcsv(joinpath(datapath,"gamlr.$family.$fitname.coefs.CVmin.csv"))))
                    gcoefs_CV1se = vec(convert(Matrix{Float64},readcsv(joinpath(datapath,"gamlr.$family.$fitname.coefs.CV1se.csv"))))

                    glp_CVmin = coef(glp,select=:CVmin,nCVfolds=10)
                    glp_CV1se = coef(glp,select=:CV1se,nCVfolds=10)

                    @fact glp_CVmin --> roughly(gcoefs_CVmin;rtol=0.3)
                    @fact glp_CV1se --> roughly(gcoefs_CV1se;rtol=0.3)

                    if γ==0
                        # Compare with LassoPath
                        lp = fit(LassoPath, X, y, dist, link; λminratio=0.001) #, λ=λ)
                        @fact glp.λ --> lp.λ
                        @fact glp.b0 --> lp.b0
                        @fact glp.coefs --> lp.coefs
                    end
                end
            end
        end
    end
end

## the following code is useful for understanding comparison failures

# rdist(x::Number,y::Number) = abs(x-y)/max(abs(x),abs(y))
# rdist{T<:Number,S<:Number}(x::AbstractArray{T}, y::AbstractArray{S}; norm::Function=vecnorm) = norm(x - y) / max(norm(x), norm(y))
#
# (family, dist, link) = (("gaussian", Normal(), IdentityLink()), ("binomial", Binomial(), LogitLink()), ("poisson", Poisson(), LogLink()))[3]
# data = readcsv(joinpath(datapath,"gamlr.$family.data.csv"))
# y = data[:,1]
# X = data[:,2:end]
# (n,p) = size(X)
# γ = [0 2 10][1]
# fitname = "gamma$γ"
# # get gamlr params and estimates
# params = readtable(joinpath(datapath,"gamlr.$family.$fitname.params.csv"))
# fittable = readtable(joinpath(datapath,"gamlr.$family.$fitname.fit.csv"))
# gcoefs = convert(Matrix{Float64},readcsv(joinpath(datapath,"gamlr.$family.$fitname.coefs.csv")))
# family = params[1,:fit_family]
# λ = nothing #convert(Vector{Float64},fittable[:fit_lambda]) # should be set to nothing evenatually
# # fit julia version
# glp = fit(GammaLassoPath, X, y, dist, link, λ=λ,γ=γ,standardize=true, λminratio=0.001)
#
# # compare
# @fact issimilarhead(glp.λ,fittable[:fit_lambda];rtol=rtol) --> true
# @fact issimilarhead(glp.b0,fittable[:fit_alpha];rtol=rtol) --> true
# @fact issimilarhead(full(glp.coefs'),gcoefs';rtol=rtol) --> true
# # we follow GLM.jl convention where deviance is scaled by nobs, while in gamlr it is not
# @fact issimilarhead(deviance(glp),fittable[:fit_deviance]/nobs(glp);rtol=rtol) --> true
# @fact issimilarhead(deviance(glp,X,y),fittable[:fit_deviance]/nobs(glp);rtol=rtol) --> true
# # @fact issimilarhead(round(df(glp)[2:end]),round(fittable[2:end,:fit_df])) --> true
# @fact issimilarhead(loglikelihood(glp),fittable[:fit_logLik];rtol=rtol) --> true
# @fact issimilarhead(aicc(glp),fittable[:fit_AICc];rtol=rtol) --> true
# # what we really need all these stats for is that the AICc identifies the same minima:
# if indmin(aicc(glp)) != endof(aicc(glp)) && indmin(fittable[:fit_AICc]) != endof(fittable[:fit_AICc])
#     # interior minima
#     @fact indmin(aicc(glp)) --> indmin(fittable[:fit_AICc])
# end
#
# # comparse CV
# gcoefs_CVmin = vec(convert(Matrix{Float64},readcsv(joinpath(datapath,"gamlr.$family.$fitname.coefs.CVmin.csv"))))
# gcoefs_CV1se = vec(convert(Matrix{Float64},readcsv(joinpath(datapath,"gamlr.$family.$fitname.coefs.CV1se.csv"))))
#
# glp_CVmin = coef(glp,select=:CVmin,nCVfolds=10)
# glp_CV1se = coef(glp,select=:CV1se,nCVfolds=10)
#
# @fact glp_CVmin --> roughly(gcoefs_CVmin;rtol=0.3)
# @fact glp_CV1se --> roughly(gcoefs_CV1se;rtol=0.3)

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
# lp = fit(LassoPath, X, y, dist, link; λ=λ, λminratio=0.001) #, λminratio=0.001)
# @fact glp.λ --> lp.λ
# @fact glp.b0 --> lp.b0
# @fact glp.coefs --> lp.coefs
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
