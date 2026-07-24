using JSON

# extra distributions used only by the JSON serialization tests below, generated
# with a mean-matching random draw (mirrors the `randdist` pattern in lasso.jl)
randdist(::Gamma, x) = rand(testrng, Gamma(2.0, x / 2))
randdist(::InverseGaussian, x) = rand(testrng, InverseGaussian(x, 1.0))
randdist(::NegativeBinomial, x) = rand(testrng, NegativeBinomial(10.0, 10.0 / (10.0 + x)))

"""
Check that `m`'s prediction matches after roundtripping through `to_dict`
(JSON-independent) and through `write_json`/`read_json` (JSON-based).
"""
function test_json_roundtrip(m, X; offset::AbstractVector{<:Real}=Float64[])
    expected = isempty(offset) ? predict(m, X) : predict(m, X; offset=offset)

    d = Lasso.to_dict(m)
    m_dict = Lasso.InferenceModel(d)
    actual_dict = isempty(offset) ? predict(m_dict, X) : predict(m_dict, X; offset=offset)
    @test actual_dict ≈ expected

    io = IOBuffer()
    write_json(io, m)
    m_json = read_json(IOBuffer(take!(io)))
    actual_json = isempty(offset) ? predict(m_json, X) : predict(m_json, X; offset=offset)
    @test actual_json ≈ expected

    return d
end

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

    @testset "distribution/link coverage" begin
        Random.seed!(testrng, 371)
        (X, _) = genrand(Float64, Normal(), IdentityLink(), 200, 5, false)
        beta = [(-1)^j * exp(-2 * (j - 1) / 20) for j = 1:5]
        eta = 0.3 .* (X * beta)

        # generate y in a domain-valid way *independent* of the link under test
        # (fitting with a non-canonical link doesn't require y to have been
        # generated through that link's inverse)
        positive_mu = exp.(eta)
        unit_mu = GLM.linkinv.(LogitLink(), eta)

        @testset "$(typeof(dist).name.name) $(typeof(link).name.name)" for (dist, link, mu) in (
            (Poisson(), LogLink(), positive_mu),
            (Gamma(), InverseLink(), positive_mu),
            (Gamma(), LogLink(), positive_mu),
            (InverseGaussian(), InverseSquareLink(), positive_mu),
            (NegativeBinomial(10.0, 0.5), LogLink(), positive_mu),
            (Binomial(), ProbitLink(), unit_mu),
            (Binomial(), CloglogLink(), unit_mu),
        )
            y = [randdist(dist, m) for m in mu]
            m = fit(LassoModel, X, Float64.(y), dist, link; select=MinAICc())

            d = test_json_roundtrip(m, X)
            @test d["link"] == string(nameof(typeof(link)))
        end
    end

    @testset "weights" begin
        Random.seed!(testrng, 371)
        (X, y) = genrand(Float64, Normal(), IdentityLink(), 200, 5, false)
        wts = rand(testrng, 200) .+ 0.5

        m = fit(LassoModel, X, y; wts=wts, select=MinAICc())
        test_json_roundtrip(m, X)
    end

    @testset "penalty_factor" begin
        Random.seed!(testrng, 371)
        (X, y) = genrand(Float64, Normal(), IdentityLink(), 200, 5, false)
        penalty_factor = [0.0, 1.0, 1.0, 0.5, 2.0]

        m = fit(LassoModel, X, y; penalty_factor=penalty_factor, select=MinAICc())
        test_json_roundtrip(m, X)
    end

    @testset "sparse X" begin
        Random.seed!(testrng, 371)
        (X, y) = genrand(Float64, Normal(), IdentityLink(), 200, 5, true)

        m = fit(LassoModel, sparse(X), y; select=MinAICc())
        test_json_roundtrip(m, Matrix(X))
    end

    @testset "segment selectors" begin
        Random.seed!(testrng, 371)
        (X, y) = genrand(Float64, Normal(), IdentityLink(), 200, 5, false)
        path = fit(LassoPath, X, y)

        @testset "$(typeof(select))" for select in
                (MinAIC(), MinAICc(), MinBIC(), MinCVmse(path), MinCV1se(path))
            Random.seed!(421)
            m = fit(LassoModel, X, y; select=select)
            test_json_roundtrip(m, X)
        end
    end

    @testset "GammaLassoModel" begin
        Random.seed!(testrng, 371)
        (X, y) = genrand(Float64, Normal(), IdentityLink(), 200, 5, false)

        m = fit(GammaLassoModel, X, y; γ=1.0, select=MinAICc())
        test_json_roundtrip(m, X)
    end

    @testset "offset" begin
        Random.seed!(testrng, 371)
        (X, y) = genrand(Float64, Poisson(), LogLink(), 200, 5, false)
        offset = fill(0.1, length(y))

        m = fit(LassoModel, X, y, Poisson(), LogLink(); offset=offset, select=MinAICc())
        d = test_json_roundtrip(m, X; offset=offset)
        @test d["hasoffset"]

        m2 = Lasso.InferenceModel(d)
        @test_throws ArgumentError predict(m2, X)
        @test_throws ArgumentError predict(m2, X; offset=offset[1:end-1])

        m0 = fit(LassoModel, X, y, Poisson(), LogLink(); select=MinAICc())
        d0 = Lasso.to_dict(m0)
        @test !d0["hasoffset"]
        m0_2 = Lasso.InferenceModel(d0)
        @test_throws ArgumentError predict(m0_2, X; offset=offset)
    end

    @testset "unsupported model types" begin
        Random.seed!(testrng, 371)
        (X, y) = genrand(Float64, Normal(), IdentityLink(), 50, 3, false)
        path = fit(LassoPath, X, y)
        @test_throws ArgumentError Lasso.to_dict(path)

        gpath = fit(GammaLassoPath, X, y)
        @test_throws ArgumentError Lasso.to_dict(gpath)

        fl = fit(FusedLasso, y, 1.0)
        @test_throws ArgumentError Lasso.to_dict(fl)

        tf = fit(TrendFilter, y, 1, 1.0)
        @test_throws ArgumentError Lasso.to_dict(tf)
    end

    @testset "TableRegressionModel wrapper" begin
        data = DataFrame(X1=randn(testrng, 100), X2=randn(testrng, 100))
        data.Y = 1 .+ 2 .* data.X1 .- data.X2 .+ 0.1 .* randn(testrng, 100)

        m = fit(LassoModel, @formula(Y ~ X1 + X2), data; select=MinAICc())
        newX = Matrix(data[:, [:X1, :X2]])

        d = Lasso.to_dict(m)
        m2 = Lasso.InferenceModel(d)
        @test predict(m, data) ≈ predict(m2, newX)

        mktempdir() do dir
            path = joinpath(dir, "model.json")
            write_json(path, m)
            m3 = read_json(path)
            @test predict(m, data) ≈ predict(m3, newX)
        end
    end

    @testset "serialization stability" begin
        # This Dict/JSON is a hand-written, frozen stand-in for a model
        # serialized by an older version of Lasso.jl (it is not produced by
        # fitting a model here). This test should NOT fail in a non-breaking
        # release: its purpose is to make sure models serialized by older
        # versions of the package can still be deserialized and used for
        # inference. If this test fails, check whether the change is an
        # intentional, documented breaking change to the serialization format.
        d = Dict(
            "coef" => [1.0, 2.0, -1.0],
            "intercept" => true,
            "link" => "IdentityLink",
            "hasoffset" => false,
        )
        json_str = """{"coef":[1.0,2.0,-1.0],"intercept":true,"link":"IdentityLink","hasoffset":false}"""

        X = [1.0 2.0; 3.0 -1.0; 0.0 0.0]
        expected = [1.0, 8.0, 1.0]

        m_dict = Lasso.InferenceModel(d)
        @test predict(m_dict, X) ≈ expected

        m_json = read_json(IOBuffer(json_str))
        @test predict(m_json, X) ≈ expected
    end
end
