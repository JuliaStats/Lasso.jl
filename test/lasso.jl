using Lasso, GLM, Distributions, GLMNet, FactCheck

datapath = joinpath(dirname(@__FILE__), "data")

testpath(T::DataType, d::Normal, l::GLM.Link, nsamples::Int, nfeatures::Int) =
    joinpath(datapath, "$(T)_$(typeof(d).name.name)_$(typeof(l).name.name)_$(nsamples)_$(nfeatures).tsv")

function makeX(ρ, nsamples, nfeatures, sparse)
    Σ = fill(ρ, nfeatures, nfeatures)
    Σ[diagind(Σ)] = 1
    X = rand(MvNormal(Σ), nsamples)'
    sparse && (X[randperm(length(X))[1:round(Int, length(X)*0.95)]] = 0)
    β = [(-1)^j*exp(-2*(j-1)/20) for j = 1:nfeatures]
    (X, β)
end

randdist(::Normal, x) = rand(Normal(x))
randdist(::Binomial, x) = rand(Bernoulli(x))
randdist(::Poisson, x) = rand(Poisson(x))
function genrand(T::DataType, d::Distribution, l::GLM.Link, nsamples::Int, nfeatures::Int, sparse::Bool)
    X, coef = makeX(0.0, nsamples, nfeatures, sparse)
    y = X*coef
    for i = 1:length(y)
        y[i] = randdist(d, linkinv(l, y[i]))
    end
    (X, y)
end

function gen_penalty_factors(X,nonone_penalty_factors;sparcity=0.7)
    if nonone_penalty_factors
        penalty_factor = ones(size(X,2))
        nonone = Int(floor(size(X,2)*(1-sparcity)))
        srand(7337)
        penalty_factor[1:nonone] = rand(Float64,nonone)
        penalty_factor_glmnet = penalty_factor
    else
        penalty_factor = nothing
        penalty_factor_glmnet = ones(size(X,2))
    end
    penalty_factor, penalty_factor_glmnet
end

# Test against GLMNet
facts("LassoPath") do
    for (dist, link) in ((Normal(), IdentityLink()), (Binomial(), LogitLink()), (Poisson(), LogLink()))
        context("$(typeof(dist).name.name) $(typeof(link).name.name)") do
            for sp in (false, true)
                srand(1337)
                context(sp ? "sparse" : "dense") do
                    (X, y) = genrand(Float64, dist, link, 1000, 10, sp)
                    yoff = randn(length(y))
                    for intercept = (false, true)
                        context("$(intercept ? "w/" : "w/o") intercept") do
                            for alpha = [1, 0.5]
                                context("alpha = $alpha") do
                                    for nonone_penalty_factors in (false,true)
                                        context("$(nonone_penalty_factors ? "non-one" : "all-one") penalty factors") do
                                            penalty_factor, penalty_factor_glmnet = gen_penalty_factors(X,nonone_penalty_factors)
                                            for offset = Vector{Float64}[Float64[], yoff]
                                                context("$(isempty(offset) ? "w/o" : "w/") offset") do
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

                                                        for randomize = [false, true]
                                                            context(randomize ? "random" : "sequential") do
                                                                niter = 0
                                                                for algorithm = [NaiveCoordinateDescent, CovarianceCoordinateDescent]
                                                                     context(algorithm == NaiveCoordinateDescent ? "naive" : "covariance") do
                                                                         for spfit in (true,false)
                                                                             context(spfit ? "as SparseMatrixCSC" : "as Matrix") do
                                                                                criterion = :coef
                                                                                #  for criterion in (:coef,:obj) # takes too long for travis
                                                                                #      context("criterion = $criterion") do
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
                                                                                        α=alpha, offset=offset, penalty_factor=penalty_factor)
                                                                                rd = (l.coefs - gbeta)./gbeta
                                                                                rd[!isfinite(rd)] = 0
                                                                                println("         coefs adiff = $(maxabs(l.coefs - gbeta)) rdiff = $(maxabs(rd))")
                                                                                rd = (l.b0 - g.a0)./g.a0
                                                                                rd[!isfinite(rd)] = 0
                                                                                println("         b0    adiff = $(maxabs(l.b0 - g.a0)) rdiff = $(maxabs(rd))")
                                                                                if criterion==:obj
                                                                                    # nothing to compare results against at this point, we just make sure the code runs
                                                                                else
                                                                                    # @fact l.λ --> roughly(g.lambda, 5e-7)
                                                                                    @fact l.coefs --> roughly(gbeta, 5e-7)
                                                                                    @fact l.b0 --> roughly(g.a0, 2e-5)

                                                                                    # Ensure same number of iterations with all algorithms
                                                                                    if niter == 0
                                                                                        niter = l.niter
                                                                                    else
                                                                                        @fact abs(niter - l.niter) --> less_than_or_equal(10)
                                                                                    end
                                                                                end
                                                                                #     end
                                                                                # end
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
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

# Test for sparse matrices
