# Comparing with Matt Taddy's gamlr.R
# To rebuild the test cases source(gammalasso.R)
using Lasso
using GLM, FactCheck, DataFrames, Gadfly

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
plotspath = joinpath(dirname(@__FILE__), "plots")
mkpath(plotspath)

rtol=1e-2
facts("GammaLassoPath") do
    for (family, dist, link) in (("gaussian", Normal(), IdentityLink()), ("binomial", Binomial(), LogitLink()), ("poisson", Poisson(), LogLink()))
        context(family) do
            data = readcsv(joinpath(datapath,"gamlr.$family.data.csv"))
            y = data[:,1]
            X = data[:,2:end]
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
                    # what we really need all these stats for is that the AICc identifies the same minima:
                    if indmin(aicc(glp)) != endof(aicc(glp)) && indmin(fittable[:fit_AICc]) != endof(fittable[:fit_AICc])
                        # interior minima
                        @fact indmin(aicc(glp)) --> indmin(fittable[:fit_AICc])
                    end

                    if γ==0
                        # Compare with LassoPath
                        lp = fit(LassoPath, X, y, dist, link; λminratio=0.001) #, λ=λ)
                        @fact glp.λ --> lp.λ
                        @fact glp.b0 --> lp.b0
                        @fact glp.coefs --> lp.coefs
                    end

                    # test plots
                    p = plot(glp)
                    filename = joinpath(plotspath,"$family.$fitname.path.svg")
                    draw(SVG(filename,5inch,5inch),p)
                    @fact isfile(filename) --> true
                end
            end
        end
    end
end

# comment the next line to see plots after test finishes in Lasso/test/plots/
rm(plotspath;recursive=true)

## the following code is useful for understanding comparison failures

(family, dist, link) = (("gaussian", Normal(), IdentityLink()), ("binomial", Binomial(), LogitLink()), ("poisson", Poisson(), LogLink()))[1]
data = readcsv(joinpath(datapath,"gamlr.$family.data.csv"))
y = data[:,1]
X = data[:,2:end]
(n,p) = size(X)
γ = [0 2 10][1]
fitname = "gamma$γ"
# get gamlr params and estimates
params = readtable(joinpath(datapath,"gamlr.$family.$fitname.params.csv"))
fittable = readtable(joinpath(datapath,"gamlr.$family.$fitname.fit.csv"))
gcoefs = convert(Matrix{Float64},readcsv(joinpath(datapath,"gamlr.$family.$fitname.coefs.csv")))
family = params[1,:fit_family]
λ = nothing #convert(Vector{Float64},fittable[:fit_lambda]) # should be set to nothing evenatually
# fit julia version
glp = fit(GammaLassoPath, X, y, dist, link, λ=λ,γ=γ,standardize=true, λminratio=0.001)

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
