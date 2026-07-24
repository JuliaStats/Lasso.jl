using JSON

@testset "JSON roundtrip" begin
    @testset "$(typeof(dist).name.name) $(typeof(link).name.name)" for (dist, link) in
            ((Normal(), IdentityLink()), (Binomial(), LogitLink()))
        Random.seed!(testrng, 371)
        (X, y) = genrand(Float64, dist, link, 200, 5, false)

        @testset "$(intercept ? "w/" : "w/o") intercept" for intercept in (true, false)
            m = fit(LassoModel, X, y, dist, link; intercept=intercept, select=MinAICc())

            @testset "path" begin
                mktempdir() do dir
                    path = joinpath(dir, "model.json")
                    write_json(path, m)
                    m2 = read_json(path)

                    @test m2 isa Lasso.InferenceModel
                    @test predict(m, X) ≈ predict(m2, X)
                end
            end

            @testset "IO" begin
                io = IOBuffer()
                write_json(io, m)
                m2 = read_json(IOBuffer(take!(io)))

                @test m2 isa Lasso.InferenceModel
                @test predict(m, X) ≈ predict(m2, X)
            end

            @testset "to_dict (JSON-independent)" begin
                d = Lasso.to_dict(m)
                m2 = Lasso.InferenceModel(d)

                @test m2 isa Lasso.InferenceModel
                @test predict(m, X) ≈ predict(m2, X)
            end
        end
    end
end
