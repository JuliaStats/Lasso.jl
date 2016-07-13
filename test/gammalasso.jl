# using Lasso
using GLM, Distributions, GLMNet, FactCheck, RCall, DataFrames

import StatsBase.deviance, StatsBase.nobs
nobs(path::RegularizationPath) = length(path.m.rr.y)
deviance(path::RegularizationPath) = (1 .- path.pct_dev) .* (path.nulldev * nobs(path))
function issimilarhead(a::AbstractVector,b::AbstractVector;rtol=1e-4)
    n = min(size(a,1),size(b,1))
    isapprox(a[1:n],b[1:n];rtol=rtol) #? true : [a[1:n] b[1:n]]
end
function issimilarhead(a::AbstractMatrix,b::AbstractMatrix;rtol=1e-4)
    n = min(size(a,1),size(b,1))
    m = min(size(a,2),size(b,2))
    isapprox(a[1:n,1:m],b[1:n,1:m];rtol=rtol) #? true : [a[1:n,1:m] b[1:n,1:m]]
end
datapath = joinpath(dirname(@__FILE__), "data")

# fitname="fitgl"
# params = readtable(joinpath(datapath,"gamlr_$fitname.params.csv"))
# fittable = readtable(joinpath(datapath,"gamlr_$fitname.csv"))
# coefs = readcsv(joinpath(datapath,"gamlr_$fitname.coefs.csv"))
# family = params[1,:fit_family]
# γ=params[1,:fit_gamma]
# l = fit(GammaLassoPath, X, y; λminratio=0.001,γ=γ)
# nλ = length(l.λ)
# @fact deviance(l) --> roughly(fittable[1:nλ,:fit_deviance], rtol)
# [deviance(l) fittable[1:nλ,:fit_deviance]]
# [l.coefs' coefs[:,1:nλ]']
# issimilarhead(l.coefs',coefs';rtol=0.001)
#
# coefs'
# l.coefs'

rtol=1e-2
facts("GammaLassoPath") do
    for (family, dist, link) in (("gaussian", Normal(), IdentityLink()), ("binomial", Binomial(), LogitLink()), ("poisson", Poisson(), LogLink()))[1:3]
        context(family) do
            data = readcsv(joinpath(datapath,"gamlr.$family.data.csv"))
            y = data[:,1]
            X = data[:,2:end]
            (n,p) = size(X)
            for fitname in ["fitlasso", "fitgl", "fitglbv"][1:3]
                # get gamlr params and estimates
                params = readtable(joinpath(datapath,"gamlr.$family.$fitname.params.csv"))
                fittable = readtable(joinpath(datapath,"gamlr.$family.$fitname.fit.csv"))
                gcoefs = convert(Matrix{Float64},readcsv(joinpath(datapath,"gamlr.$family.$fitname.coefs.csv")))
                family = params[1,:fit_family]
                γ=params[1,:fit_gamma]
                context("$fitname: γ=$γ, family=$family") do

                    # fit julia version
                    l = fit(GammaLassoPath, X, y; λminratio=0.001,γ=γ)

                    # compare
                    @fact issimilarhead(l.λ,fittable[:fit_lambda];rtol=rtol) --> true
                    @fact issimilarhead(l.b0,fittable[:fit_alpha];rtol=rtol) --> true
                    @fact issimilarhead(full(l.coefs'),gcoefs';rtol=rtol) --> true
                    # @fact issimilarhead(deviance(l),fittable[:fit_deviance];rtol=rtol*10) --> true

                    if γ==0
                        context("γ=0 (Lasso), try to compare with GLMNet") do
                            if isa(dist, Normal)
                                yp = y
                                # ypstd = std(yp, corrected=false)
                                # # glmnet does this on entry, which changes λ mappings, but not
                                # # coefficients. Should we?
                                # yp ./= ypstd
                                g = glmnet(X, yp, dist; lambda_min_ratio=0.001, tol=10*eps())
                            elseif isa(dist, Binomial)
                                yp = zeros(size(y, 1), 2)
                                yp[:, 1] = y .== 0
                                yp[:, 2] = y .== 1
                                g = glmnet(X, yp, dist; lambda_min_ratio=0.001, tol=10*eps())
                            else
                                g = glmnet(X, y, dist; lambda_min_ratio=0.001, tol=10*eps())
                            end
                            gbeta = convert(Matrix{Float64}, g.betas)

                            @fact issimilarhead(l.λ,g.lambda;rtol=rtol) --> true
                            @fact issimilarhead(l.b0,g.a0;rtol=rtol) --> true
                            @fact issimilarhead(full(l.coefs'),gbeta';rtol=rtol) --> true
                            # @fact l.λ --> roughly(g.lambda; rtol=rtol)
                            # @fact full(l.coefs) --> roughly(gbeta; rtol=rtol)
                            # @fact l.b0 --> roughly(g.a0; rtol=rtol)
                        end
                    end
                end
            end
        end
    end
end
