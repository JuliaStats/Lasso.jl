using Random
@testset "segment/model selection" begin

datapath = joinpath(dirname(@__FILE__), "data")

@testset "$family" for (family, dist, link) in (("gaussian", Normal(), IdentityLink()), ("binomial", Binomial(), LogitLink()), ("poisson", Poisson(), LogLink()))
    data = readcsvmat(joinpath(datapath,"gamlr.$family.data.csv"))
    y = convert(Vector{Float64},data[:,1])
    X = convert(Matrix{Float64},data[:,2:end])
    Xwconst = [ones(size(X,1),1) X]
    offset = fill(0.001,size(y))

    @testset "$L" for L in [LassoModel, GammaLassoModel]
        R = Lasso.pathtype(L)
        path = fit(R, X, y, dist, link; offset=offset)

        @testset "$(typeof(select))" for select in [MinAIC(), MinAICc(), MinBIC(), MinCVmse(path), MinCV1se(path)]
            Random.seed!(421)
            m = fit(L, X, y, dist, link; select=select, offset=offset);

            Random.seed!(421)
            pathcoefs = coef(path, select)

            Random.seed!(421)
            pathpredict = Lasso.predict(path, X; select=select, offset=offset)

            @test pathcoefs == coef(m)
            if isa(m, LinearModel)
                @test pathpredict == GLM.predict(m, Xwconst) + offset
            else
                @test pathpredict == GLM.predict(m, Xwconst; offset=offset)
            end
        end
    end
end

end
