using Lasso
using GLM, Distributions, Random, LinearAlgebra, SparseArrays
using GLMNet: glmnet

datapath = joinpath(dirname(@__FILE__), "data")

testpath(T::DataType, d::Normal, l::GLM.Link, nsamples::Int, nfeatures::Int) =
    joinpath(datapath, "$(T)_$(typeof(d).name.name)_$(typeof(l).name.name)_$(nsamples)_$(nfeatures).tsv")

function makeX(ρ, nsamples, nfeatures, sparse)
    Σ = fill(ρ, nfeatures, nfeatures)
    Σ[diagind(Σ)] .= 1
    X = permutedims(rand(testrng, MvNormal(Σ), nsamples))
    sparse && (X[randperm(testrng, length(X))[1:round(Int, length(X)*0.95)]] .= 0)
    β = [(-1)^j*exp(-2*(j-1)/20) for j = 1:nfeatures]
    (X, β)
end

randdist(::Normal, x) = rand(testrng, Normal(x))
randdist(::Binomial, x) = rand(testrng, Bernoulli(x))
randdist(::Poisson, x) = rand(testrng, Poisson(x))
function genrand(T::DataType, d::Distribution, l::GLM.Link, nsamples::Int, nfeatures::Int, sparse::Bool)
    X, coef = makeX(0.0, nsamples, nfeatures, sparse)
    y = X*coef
    for i = 1:length(y)
        y[i] = randdist(d, GLM.linkinv(l, y[i]))
    end
    (X, y)
end

function gen_penalty_factors(X,nonone_penalty_factors;frac1=0.7,frac0=0.05)
    if nonone_penalty_factors
        penalty_factor = ones(size(X,2))
        nzeros = Int(floor(size(X,2)*frac0))
        nonone = Int(floor(size(X,2)*(1-frac1)))
        Random.seed!(testrng, 7337)
        penalty_factor[1:nzeros] = zeros(nzeros)
        penalty_factor[end-nonone+1:end] = rand(testrng, Float64,nonone)
        penalty_factor_glmnet = penalty_factor
    else
        penalty_factor = nothing
        penalty_factor_glmnet = ones(size(X,2))
    end
    penalty_factor, penalty_factor_glmnet
end

# Test against GLMNet
@testset "LassoPath" begin
    @testset "$(typeof(dist).name.name) $(typeof(link).name.name)" for (dist, link) in ((Normal(), IdentityLink()), (Binomial(), LogitLink()), (Poisson(), LogLink()))
        @testset "sparse = $sp" for sp in (false, true)
            Random.seed!(testrng, 465)
            (X, y) = genrand(Float64, dist, link, 1000, 10, sp)
            yoff = randn(testrng, length(y))
            @testset "$(intercept ? "w/" : "w/o") intercept" for intercept = (false, true)
                @testset "unfitted LassoPath (dofit=false)" begin
                    @testset "$(spfit ? "as SparseMatrixCSC" : "as Matrix")" for spfit in (true,false)
                        l = fit(LassoPath, spfit ? sparse(X) : X, y, dist, link,
                            intercept=intercept, dofit=false)

                        p = size(X,2)
                        if intercept
                            p += 1
                        end
                        @test size(l) == (p,1)
                        @test coef(l) == zeros(eltype(X),p,1)
                    end
                end

                @testset "alpha = $alpha" for alpha = (1, 0.5)
                    @testset "$(nonone_penalty_factors ? "non-one" : "all-one") penalty factors" for nonone_penalty_factors in (false,true)
                        penalty_factor, penalty_factor_glmnet = gen_penalty_factors(X,nonone_penalty_factors)
                        @testset "$(isempty(offset) ? "w/o" : "w/") offset" for offset = Vector{Float64}[Float64[], yoff]
                            let y=y
                                # First fit with GLMNet
                                if isa(dist, Normal)
                                    yp = isempty(offset) ? y : y + offset
                                    ypstd = std(yp, corrected=false)
                                    # glmnet does this on entry, which changes λ mappings, but not
                                    # coefficients. Should we?
                                    yp = yp ./ ypstd
                                    !isempty(offset) && (offset = offset ./ ypstd)
                                    y = y ./ ypstd
                                    g = glmnet(X, yp, dist, intercept=intercept, alpha=alpha, tol=10*eps(); penalty_factor=penalty_factor_glmnet)
                                elseif isa(dist, Binomial)
                                    yp = zeros(size(y, 1), 2)
                                    yp[:, 1] = y .== 0
                                    yp[:, 2] = y .== 1
                                    g = glmnet(X, yp, dist, intercept=intercept, alpha=alpha, tol=10*eps(),
                                               offsets=isempty(offset) ? zeros(length(y)) : offset; penalty_factor=penalty_factor_glmnet)
                                else
                                    g = glmnet(X, y, dist, intercept=intercept, alpha=alpha, tol=10*eps(),
                                               offsets=isempty(offset) ? zeros(length(y)) : offset; penalty_factor=penalty_factor_glmnet)
                                end
                                gbeta = convert(Matrix{Float64}, g.betas)

                                @testset "$(randomize ? "random" : "sequential")" for randomize = [false, true]
                                    niter = 0
                                    @testset "$(algorithm == NaiveCoordinateDescent ? "naive" : "covariance")" for algorithm = (NaiveCoordinateDescent, CovarianceCoordinateDescent)
                                        @testset "$(spfit ? "as SparseMatrixCSC" : "as Matrix")" for spfit in (true,false)
                                            criterion = :coef
                                            #  for criterion in (:coef,:obj) # takes too long for travis
                                            #      @testset "criterion = $criterion" begin
                                            if criterion == :obj
                                                irls_tol = 100*eps()
                                                cd_tol = 100*eps()
                                            else
                                                irls_tol = 10*eps()
                                                cd_tol = 10*eps()
                                            end
                                            # Now fit with Lasso
                                            l = fit(LassoPath, spfit ? sparse(X) : X, y, dist, link,
                                                    λ=g.lambda, algorithm=algorithm, intercept=intercept,
                                                    cd_tol=cd_tol, irls_tol=irls_tol, criterion=criterion, randomize=randomize,
                                                    rng=StableRNG(1337),
                                                    α=alpha, offset=offset, penalty_factor=penalty_factor)
                                            rd = (l.coefs - gbeta)./gbeta
                                            rd[.!isfinite.(rd)] .= 0
                                            println("         coefs adiff = $(maximum(abs, l.coefs - gbeta)) rdiff = $(maximum(abs, rd))")
                                            rd = (l.b0 - g.a0)./g.a0
                                            rd[.!isfinite.(rd)] .= 0
                                            println("         b0    adiff = $(maximum(abs, l.b0 - g.a0)) rdiff = $(maximum(abs, rd))")
                                            if criterion==:obj
                                                # nothing to compare results against at this point, we just make sure the code runs
                                            else
                                                # @test l.λ ≈ g.lambda rtol=5e-7
                                                @test l.coefs ≈ gbeta rtol=5e-7
                                                @test l.b0 ≈ g.a0 rtol=2e-5

                                                # Ensure same number of iterations with all algorithms
                                                if niter == 0
                                                    niter = l.niter
                                                else
                                                    @test abs(niter - l.niter) <= 10
                                                end
                                            end
                                            #     end
                                            # end
                                            # With a function to generate λ
                                            λfunc(λmax) = range(λmax, stop=g.lambda[end], length=10)
                                            lf = fit(LassoPath, spfit ? sparse(X) : X, y, dist, link,
                                                     λ=λfunc, algorithm=algorithm, intercept=intercept,
                                                     cd_tol=cd_tol, irls_tol=irls_tol, criterion=criterion, randomize=randomize,
                                                     rng=StableRNG(1337),
                                                     α=alpha, offset=offset, penalty_factor=penalty_factor)
                                            @test length(lf.λ) == 10
                                            @test_skip all(iszero, lf.coefs[:,1])
                                            @test lf.coefs[:,end] ≈ l.coefs[:,end]
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

# Test case with zero variation in y is handled correctly
function zero_variation_test()
    X = [
        0.5472502169628388 0.37660447632078875 0.06669114126498532 0.4950818154768257;
        0.5142931961160688 0.520205941129849 0.4052730635141131 0.6700530909562794;
        0.5831846867316071 0.3174143498124731 0.772131243876973 0.03386847158881201;
        0.8802489459954292 0.6742158685234003 0.3849775799923969 0.7773264968613842;
        0.9216786846192617 0.7888303438159934 0.09788865152005011 0.34950775139369905
    ]
    y = 0.2937233091452627 .+ zeros(size(X, 1))
    path = fit(LassoPath, X, y)
    (path.λ == eltype(path.λ)[0]) || return false
    (length(path.coefs.nzval) == 0) || return false
    return true
end

@test zero_variation_test() == true

# Test for sparse matrices

# @testset "LassoPath Zero in" begin
#     for (dist, link) in ((Normal(), IdentityLink()), (Binomial(), LogitLink()), (Poisson(), LogLink()))[2:2]
#         @testset "$(typeof(dist).name.name) $(typeof(link).name.name)" begin
#             for sp in (false, true)[2:2]
#                 Random.seed!(testrng, 1337)
#                 @testset sp ? "sparse" : "dense" begin
#                     (X, y) = genrand(Float64, dist, link, 1000, 10, sp)
#                     yoff = randn(testrng, length(y))
#                     for intercept = (false, true)[2:2]
#                         @testset "$(intercept ? "w/" : "w/o") intercept" begin
#                             @testset "unfitted LassoPath (dofit=false)" begin
#                                 for spfit in (true,false)
#                                     @testset spfit ? "as SparseMatrixCSC" : "as Matrix" begin
#                                         l = fit(LassoPath, spfit ? sparse(X) : X, y, dist, link,
#                                             intercept=intercept, dofit=false)
#
#                                         p = size(X,2)
#                                         if intercept
#                                             p += 1
#                                         end
#                                         @test size(l) == (p,1)
#                                         @test coef(l) == zeros(eltype(X),p,1)
#                                     end
#                                 end
#                             end
#                             for alpha = [1, 0.5]
#                                 @testset "alpha = $alpha" begin
#                                     for nonone_penalty_factors in (false,true)
#                                         @testset "$(nonone_penalty_factors ? "non-one" : "all-one") penalty factors" begin
#                                             penalty_factor, penalty_factor_glmnet = gen_penalty_factors(X,nonone_penalty_factors)
#                                             for offset = Vector{Float64}[Float64[], yoff]
#                                                 @testset "$(isempty(offset) ? "w/o" : "w/") offset" begin
#                                                     let y=y
#                                                         # First fit with GLMNet
#                                                         if isa(dist, Normal)
#                                                             yp = isempty(offset) ? y : y + offset
#                                                             ypstd = std(yp, corrected=false)
#                                                             # glmnet does this on entry, which changes λ mappings, but not
#                                                             # coefficients. Should we?
#                                                             yp = yp ./ ypstd
#                                                             !isempty(offset) && (offset = offset ./ ypstd)
#                                                             y = y ./ ypstd
#                                                             g = glmnet(X, yp, dist, intercept=intercept, alpha=alpha, tol=10*eps(); penalty_factor=penalty_factor_glmnet)
#                                                         elseif isa(dist, Binomial)
#                                                             yp = zeros(size(y, 1), 2)
#                                                             yp[:, 1] = y .== 0
#                                                             yp[:, 2] = y .== 1
#                                                             g = glmnet(X, yp, dist, intercept=intercept, alpha=alpha, tol=10*eps(),
#                                                                        offsets=isempty(offset) ? zeros(length(y)) : offset; penalty_factor=penalty_factor_glmnet)
#                                                         else
#                                                             g = glmnet(X, y, dist, intercept=intercept, alpha=alpha, tol=10*eps(),
#                                                                        offsets=isempty(offset) ? zeros(length(y)) : offset; penalty_factor=penalty_factor_glmnet)
#                                                         end
#                                                         gbeta = convert(Matrix{Float64}, g.betas)
#
#                                                         for randomize = [false, true]
#                                                             @testset randomize ? "random" : "sequential" begin
#                                                                 niter = 0
#                                                                 for algorithm = [NaiveCoordinateDescent, CovarianceCoordinateDescent]
#                                                                      @testset algorithm == NaiveCoordinateDescent ? "naive" : "covariance" begin
#                                                                          for spfit in (true,false)
#                                                                              @testset spfit ? "as SparseMatrixCSC" : "as Matrix" begin
#                                                                                 criterion = :coef
#                                                                                 #  for criterion in (:coef,:obj) # takes too long for travis
#                                                                                 #      @testset "criterion = $criterion" begin
#                                                                                 if criterion == :obj
#                                                                                     irls_tol = 100*eps()
#                                                                                     cd_tol = 100*eps()
#                                                                                 else
#                                                                                     irls_tol = 10*eps()
#                                                                                     cd_tol = 10*eps()
#                                                                                 end
#                                                                                 # Now fit with Lasso
#                                                                                 l = fit(LassoPath, spfit ? sparse(X) : X, y, dist, link,
#                                                                                         λ=g.lambda, algorithm=algorithm, intercept=intercept,
#                                                                                         cd_tol=cd_tol, irls_tol=irls_tol, criterion=criterion, randomize=randomize,
#                                                                                         α=alpha, offset=offset)#, penalty_factor=penalty_factor)
#                                                                                 rd = (l.coefs - gbeta)./gbeta
#                                                                                 rd[!isfinite(rd)] = 0
#                                                                                 # println("         coefs adiff = $(maximum(abs,l.coefs - gbeta)) rdiff = $(maximum(abs,rd))")
#                                                                                 rd = (l.b0 - g.a0)./g.a0
#                                                                                 rd[!isfinite(rd)] = 0
#                                                                                 # println("         b0    adiff = $(maximum(abs,l.b0 - g.a0)) rdiff = $(maximum(abs,rd))")
#                                                                                 if criterion==:obj
#                                                                                     # nothing to compare results against at this point, we just make sure the code runs
#                                                                                 else
#                                                                                     # @test l.λ ≈ g.lambda rtol=5e-7
#                                                                                     @test l.coefs ≈ gbeta rtol=5e-7
#                                                                                     @test l.b0 ≈ g.a0 rtol=2e-5
#
#                                                                                     # Ensure same number of iterations with all algorithms
#                                                                                     if niter == 0
#                                                                                         niter = l.niter
#                                                                                     else
#                                                                                         @test abs(niter - l.niter) <= 10
#                                                                                     end
#                                                                                 end
#                                                                                 #     end
#                                                                                 # end
#                                                                             end
#                                                                         end
#                                                                     end
#                                                                 end
#                                                             end
#                                                         end
#                                                     end
#                                                 end
#                                             end
#                                         end
#                                     end
#                                 end
#                             end
#                         end
#                     end
#                 end
#             end
#         end
#     end
# end
