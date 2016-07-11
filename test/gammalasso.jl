# Test against GLMNet when γ=0
facts("GammaLassoPath") do
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
                                    for offset = Vector{Float64}[Float64[], yoff]
                                        context("$(isempty(offset) ? "w/o" : "w/") offset") do
                                            # First fit with GLMNet
                                            if isa(dist, Normal)
                                                yp = isempty(offset) ? y : y + offset
                                                ypstd = std(yp, corrected=false)
                                                # glmnet does this on entry, which changes λ mappings, but not
                                                # coefficients. Should we?
                                                yp ./= ypstd
                                                !isempty(offset) && (offset ./= ypstd)
                                                y ./= ypstd
                                                g = glmnet(X, yp, dist, intercept=intercept, alpha=alpha, tol=10*eps())
                                            elseif isa(dist, Binomial)
                                                yp = zeros(size(y, 1), 2)
                                                yp[:, 1] = y .== 0
                                                yp[:, 2] = y .== 1
                                                g = glmnet(X, yp, dist, intercept=intercept, alpha=alpha, tol=10*eps(),
                                                           offsets=isempty(offset) ? zeros(length(y)) : offset)
                                            else
                                                g = glmnet(X, y, dist, intercept=intercept, alpha=alpha, tol=10*eps(),
                                                           offsets=isempty(offset) ? zeros(length(y)) : offset)
                                            end
                                            gbeta = convert(Matrix{Float64}, g.betas)

                                            for randomize = VERSION >= v"0.4-dev+1915" ? [false, true] : [false]
                                                context(randomize ? "random" : "sequential") do
                                                    niter = 0
                                                    for naivealgorithm in (false, true)
                                                         context(naivealgorithm ? "naive" : "covariance") do
                                                             for spfit in (false, true)
                                                                 context(spfit ? "as SparseMatrixCSC" : "as Matrix") do
                                                                     for criterion in (:coef, :obj)
                                                                         context("criterion = $criterion") do
                                                                            # Now fit with GammaLasso
                                                                            l = fit(GammaLassoPath, spfit ? sparse(X) : X, y, dist, link,
                                                                                    λ=g.lambda, naivealgorithm=naivealgorithm, intercept=intercept,
                                                                                    cd_tol=10*eps(), irls_tol=10*eps(), criterion=criterion, randomize=randomize,
                                                                                    α=alpha, offset=offset)

                                                                            # rd = (l.coefs - gbeta)./gbeta
                                                                            # rd[!isfinite(rd)] = 0
                                                                            # println("         coefs adiff = $(maxabs(l.coefs - gbeta)) rdiff = $(maxabs(rd))")
                                                                            # rd = (l.b0 - g.a0)./g.a0
                                                                            # rd[!isfinite(rd)] = 0
                                                                            # println("         b0    adiff = $(maxabs(l.b0 - g.a0)) rdiff = $(maxabs(rd))")
                                                                            @fact l.coefs --> roughly(gbeta, 5e-7)
                                                                            @fact l.b0 --> roughly(g.a0, 2e-5)

                                                                            # Ensure same number of iterations with all algorithms
                                                                            if niter == 0
                                                                                niter = l.niter
                                                                            else
                                                                                @fact l.niter --> niter
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
