using Lasso
using Random, SparseArrays
DATADIR = joinpath(dirname(@__FILE__), "data")

function diffmatslow(order, n)
    D = sparse(SparseArrays.spdiagm_internal(0 => fill(-1., n-1), 1=>fill(1., n-1))...)
    for i = 1:order
        D = sparse(SparseArrays.spdiagm_internal(0 => fill(-1., n-i-1), 1=>fill(1., n-i-1))...)*D
    end
    D
end

# Test that DifferenceMatrix behaves as expected
@testset "DifferenceMatrix" begin
    @testset "order = $(order)" for order in (1, 2, 3)
        D1 = diffmatslow(order, 100)
        D2 = Lasso.TrendFiltering.DifferenceMatrix{Float64}(order, 100)
        x = randn(100)
        @test D2'D2 == D1'D1
        @test D2*x ≈ D1*x
        @test D2'*x[1:size(D1, 1)] ≈ D1'*x[1:size(D1, 1)]
    end
end

# Test against results from glmgen
lakehuron = readcsvmat(joinpath(DATADIR, "LakeHuron.csv"); header=true)[:, 3]
@testset "TrendFiltering" begin
    @testset "order = $(order)" for order in (1, 2, 3)
        @testset "lambda = $(lambda)" for lambda in (1., 10., 100.)
            @test coef(fit(TrendFilter, lakehuron, order, lambda; tol=1e-9)) ≈
                  vec(readcsvmat(joinpath(DATADIR, "LakeHuron_order_$(order)_lambda_$(lambda).csv")))
        end
    end
end
