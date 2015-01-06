module Benchmark
using Lasso, GLMNet, BenchmarkLite, Distributions

# As in Friedman et al., generate random X and Y matrix such that SNR=3
function makeXY(ρ, nsamples, nfeatures)
    Σ = fill(ρ, nfeatures, nfeatures)
    Σ[diagind(Σ)] = 1
    X = rand(MvNormal(Σ), nsamples)'
    β = [(-1)^j*exp(-2*(j-1)/20) for j = 1:nfeatures]
    signal = (β'*Σ*β)[1]
    y = X*β + scale!(randn(nsamples), sqrt(signal/3))
    # @show uvar = var(y - X*(X\y))
    # @show evar = var(y - mean(y)) - uvar
    # @show evar/uvar
    (X, y)
end

type GLMNetOp{Naive} end
calc{Naive}(::GLMNetOp{Naive}, X, y) = glmnet(X, y, naivealgorithm=Naive)

type LassoOp{Naive} end
calc{Naive}(::LassoOp{Naive}, X, y) = fit(LassoPath, X, y, naivealgorithm=Naive)

type LassoBenchmark{Op} <: Proc end
Base.length(p::LassoBenchmark, n) = 0
Base.isvalid(p::LassoBenchmark, n) = true
Base.start(p::LassoBenchmark, n) = inputs[n]
Base.run{Op}(p::LassoBenchmark{Op}, n, s) =
    calc(Op(), s[1], s[2])
Base.done(p::LassoBenchmark, n, s) = true

Base.string{Naive}(p::LassoBenchmark{GLMNetOp{Naive}}) = "glmnet $(Naive ? "naive" : "cov")"
Base.string{Naive}(p::LassoBenchmark{LassoOp{Naive}}) = "Lasso.jl $(Naive ? "naive" : "cov")"

inputs = Dict{(Float64, Int, Int),(Matrix{Float64},Vector{Float64})}()
cfgs = vec([begin
                x = (ρ, N, p)
                inputs[x] = makeXY(ρ, N, p)
                x
            end for ρ in [0, 0.1, 0.2, 0.5, 0.9, 0.95],
               (N, p) in [(1000, 100), (5000, 100), (100, 1000), (100, 5000), #=(100, 20000)=#]])
rtable = run(Proc[LassoBenchmark{GLMNetOp{true}}(), LassoBenchmark{LassoOp{true}}(),
                  LassoBenchmark{GLMNetOp{false}}(), LassoBenchmark{LassoOp{false}}()], cfgs)
show(rtable; unit=:sec)
end
