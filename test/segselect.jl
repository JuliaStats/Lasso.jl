# tests for segment selection and StatsModels @formula+df interface
using Random
@testset "segment/model selection" begin

datapath = joinpath(dirname(@__FILE__), "data")

# NOTE: 1. we use intercept=false because of a StatsModels issue where predict with drop_intercept model didn't work
# NOTE: 2. we skip the poisson because it does not converge without an intercept
@testset "$family" for (family, dist, link) in (("gaussian", Normal(), IdentityLink()), ("binomial", Binomial(), LogitLink()))
    data = CSV.read(joinpath(datapath,"gamlr.$family.data.csv"); header=[:y, :x1, :x2, :x3])
    offset = fill(0.001,length(data.y))

    @testset "$L" for L in [LassoModel, GammaLassoModel]
        R = Lasso.pathtype(L)

        @testset "$f" for (f,intercept) in ((@formula(y~x1+x2+x3), true), (@formula(y~0+x1+x2+x3), false))
            path = fit(R, f, data, dist, link; intercept=intercept, offset=offset)

            @testset "$(typeof(select))" for select in [MinAIC(), MinAICc(), MinBIC(), MinCVmse(path), MinCV1se(path)]
                Random.seed!(421)
                m = fit(L, f, data, dist, link; select=select, intercept=intercept, offset=offset)

                Random.seed!(421)
                pathcoefs = coef(path, select)

                Random.seed!(421)
                pathpredict = Lasso.predict(path, data; select=select, offset=offset)

                @test pathcoefs == coef(m)
                if isa(dist, Normal)
                    @test pathpredict == GLM.predict(m, data) + offset
                else
                    @test pathpredict == GLM.predict(m, data; offset=offset)
                end
            end
        end
    end
end

end
