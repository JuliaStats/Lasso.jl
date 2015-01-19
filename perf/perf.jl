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

type GLMNetOp{Dist,Naive} end
calc{Dist,Naive}(::GLMNetOp{Dist,Naive}, X, y) = glmnet(X, y, Dist(), naivealgorithm=Naive)
calc(::GLMNetOp{Binomial}, X, y) = glmnet(X, y, Binomial())

type LassoOp{Dist,Naive} end
calc{Dist,Naive}(::LassoOp{Dist,Naive}, X, y) = fit(LassoPath, X, y, Dist(), naivealgorithm=Naive, criterion=:coef)

type LassoBenchmark{Op} <: Proc end
Base.length(p::LassoBenchmark, n) = 0
Base.isvalid(p::LassoBenchmark, n) = true
Base.start(p::LassoBenchmark, n) = inputs[n]
function Base.start{Naive}(p::LassoBenchmark{GLMNetOp{Binomial,Naive}}, n)
    X, y = inputs[n]
    yp = zeros(length(y), 2)
    yp[:, 1] = y .== 0
    yp[:, 2] = y .== 1
    (X, yp)
end
Base.run{Op}(p::LassoBenchmark{Op}, n, s) =
    calc(Op(), s[1], s[2])
Base.done(p::LassoBenchmark, n, s) = true

Base.string{Dist,Naive}(p::LassoBenchmark{GLMNetOp{Dist,Naive}}) = "glmnet $(Dist.name.name) $(Naive ? "naive" : "cov")"
Base.string{Dist,Naive}(p::LassoBenchmark{LassoOp{Dist,Naive}}) = "Lasso.jl $(Dist.name.name) $(Naive ? "naive" : "cov")"

inputs = Dict{(Float64, Int, Int),(Matrix{Float64},Vector{Float64})}()
cfgs = vec([begin
                x = (ρ, N, p)
                inputs[x] = makeXY(ρ, N, p)
                x
            end for ρ in [0, 0.1, 0.2, 0.5, 0.9, 0.95],
               (N, p) in [(1000, 100), (5000, 100), (100, 1000), (100, 5000), #=(100, 20000)=#]])
rtable = run(Proc[LassoBenchmark{GLMNetOp{Normal,true}}(), LassoBenchmark{LassoOp{Normal,true}}(),
                  LassoBenchmark{GLMNetOp{Normal,false}}(), LassoBenchmark{LassoOp{Normal,false}}()], cfgs)
show(rtable; unit=:sec)

for (key, value) in inputs
    (X, y) = inputs[key]
    inputs[key] = (X, float64(rand(length(y)) .< 1./(1+exp(-y))))
end
rtable = run(Proc[LassoBenchmark{GLMNetOp{Binomial,true}}(), LassoBenchmark{LassoOp{Binomial,true}}()], cfgs)
show(rtable; unit=:sec)
end
