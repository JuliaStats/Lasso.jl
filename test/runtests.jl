using Lasso, GLM, Distributions, GLMNet, FactCheck

datapath = joinpath(dirname(@__FILE__), "data")

testpath(T::DataType, d::Normal, l::GLM.Link, nsamples::Int, nfeatures::Int) =
    joinpath(datapath, "$(T)_$(typeof(d).name.name)_$(typeof(l).name.name)_$(nsamples)_$(nfeatures).tsv")

function genrand(T::DataType, d::Distribution, l::GLM.Link, nsamples::Int, nfeatures::Int)
    X = randn!(Array(T, nsamples, nfeatures))
    coef = randn!(Array(T, nfeatures))
    y = linkinv!(l, Array(T, nsamples), X*coef)
    for i = 1:length(y)
        y[i] = rand((isa(d, Binomial) ? Bernoulli : typeof(d))(y[i]))
    end
    (X, y)
end

srand(1337)

for (dist, link) in ((Normal(), IdentityLink()), (Binomial(), LogitLink()), (Poisson(), LogLink()))
    facts("$(typeof(dist).name.name) $(typeof(link).name.name)") do
        (X, y) = genrand(Float64, dist, link, 1000, 10)
        for intercept = [false, true]
            context("$(intercept ? "w" : "w/o") intercept") do
                # First fit with GLMNet
                if isa(dist, Binomial)
                    yp = zeros(size(y, 1), 2)
                    yp[:, 1] = y .== 0
                    yp[:, 2] = y .== 1
                    g = glmnet(X, yp, dist, intercept=intercept)
                else
                    g = glmnet(X, y, dist, intercept=intercept)
                end
                gbeta = convert(Matrix{Float64}, g.betas)

                for naivealgorithm = [false, true]
                     context(naivealgorithm ? "naive" : "covariance") do
                        # Now fit with Lasso
                        l = fit(LassoPath, X, y, dist, link, λ=g.lambda, naivealgorithm=naivealgorithm, intercept=intercept)

                        # rd = (l.coefs - gbeta)./gbeta
                        # rd[!isfinite(rd)] = 0
                        # println("         coefs adiff = $(maxabs(l.coefs - gbeta)) rdiff = $(maxabs(rd))")
                        # rd = (l.b0 - g.a0)./g.a0
                        # rd[!isfinite(rd)] = 0
                        # println("         b0    adiff = $(maxabs(l.b0 - g.a0)) rdiff = $(maxabs(rd))")
                        @fact l.coefs => roughly(gbeta; atol=1e-3)
                        @fact l.b0 => roughly(g.a0; atol=1e-3)
                    end
                end
            end
        end
    end
end

FactCheck.exitstatus()