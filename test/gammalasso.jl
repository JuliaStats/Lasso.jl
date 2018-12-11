# Comparing with Matt Taddy's gamlr.R
# To rebuild the test cases source(gammalasso.R)
using Lasso
using CSV, GLM, DataFrames, Random, SparseArrays

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
        @testset "penalty_factor=$pf" for pf = 1:size(penaltyfactors,2)
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
                glp = fit(GammaLassoPath, X, y, dist, link; γ=γ, stopearly=false,
                    λminratio=0.001, penalty_factor=penalty_factor, λ=λ,
                    standardize=false, standardizeω=false)

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
                # if argmin(aicc(glp)) != lastindex(aicc(glp)) && argmin(fittable[Symbol("fit.AICc")]) != lastindex(fittable[Symbol("fit.AICc")])
                #     # interior minima
                #     println("comparing intereior AICc")
                #     @test argmin(aicc(glp)) == argmin(fittable[Symbol("fit.AICc")])
                # end

                # comparse CV, NOTE: this involves a random choice of train subsamples
                gcoefs_CVmin = vec(readcsvmat(joinpath(datapath,"gamlr.$family.$fitname.coefs.CVmin.csv")))
                gcoefs_CV1se = vec(readcsvmat(joinpath(datapath,"gamlr.$family.$fitname.coefs.CV1se.csv")))

                glp_CVmin = coef(glp,select=:CVmin,nCVfolds=10)
                glp_CV1se = coef(glp,select=:CV1se,nCVfolds=10)

                @test glp_CVmin ≈ gcoefs_CVmin rtol=0.3
                @test glp_CV1se ≈ gcoefs_CV1se rtol=0.3

                if γ==0
                    # Compare with LassoPath
                    lp = fit(LassoPath, X, y, dist, link; stopearly=false,
                        λminratio=0.001, penalty_factor=penalty_factor, λ=λ,
                        standardize=false, standardizeω=false)
                    @test glp.λ == lp.λ
                    @test glp.b0 ≈ lp.b0
                    @test glp.coefs ≈ lp.coefs
                end
            end
        end
    end
end
# ## the following code is useful for understanding comparison failures
#
# rdist(x::Number,y::Number) = abs(x-y)/max(abs(x),abs(y))
# rdist(x::AbstractArray{T}, y::AbstractArray{S}; norm::Function=norm) where {T<:Number,S<:Number} = norm(x - y) / max(norm(x), norm(y))
