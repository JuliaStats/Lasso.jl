module Benchmark
using Lasso, LinearAlgebra, GLMNet, BenchmarkLite, Distributions

# As in Friedman et al., generate random X and Y matrix such that SNR=3
function makeXY(ρ, nsamples, nfeatures)
    Σ = fill(ρ, nfeatures, nfeatures)
    Σ[diagind(Σ)] = 1
    X = permutedims(rand(MvNormal(Σ), nsamples))
    β = [(-1)^j*exp(-2*(j-1)/20) for j = 1:nfeatures]
    signal = (β'*Σ*β)[1]
    y = X*β + lmul!(Diagonal(randn(nsamples)), sqrt(signal/3))
    # @show uvar = var(y - X*(X\y))
    # @show evar = var(y - mean(y)) - uvar
    # @show evar/uvar
    (X, y)
end

mutable struct GLMNetOp{Dist,Naive} end
calc(::GLMNetOp{Dist,Naive}, X, y) where {Dist,Naive} = glmnet(X, y, Dist(), naivealgorithm=Naive)
calc(::GLMNetOp{Binomial}, X, y) = glmnet(X, y, Binomial())

mutable struct LassoOp{Dist,Naive} end
calc(::LassoOp{Dist,Naive}, X, y) where {Dist,Naive} = fit(LassoPath, X, y, Dist(), naivealgorithm=Naive, criterion=:coef)

mutable struct LassoBenchmark{Op} <: Proc end
Base.length(p::LassoBenchmark, n) = 0
Base.isvalid(p::LassoBenchmark, n) = true
Base.start(p::LassoBenchmark, n) = (gc(); inputs[n])
function Base.start(p::LassoBenchmark{GLMNetOp{Binomial,Naive}}, n) where Naive
    X, y = inputs[n]
    yp = zeros(length(y), 2)
    yp[:, 1] = y .== 0
    yp[:, 2] = y .== 1
    gc()
    (X, yp)
end
Base.run(p::LassoBenchmark{Op}, n, s) where {Op} =
    calc(Op(), s[1], s[2])
Base.done(p::LassoBenchmark, n, s) = true

Base.string(p::LassoBenchmark{GLMNetOp{Dist,Naive}}) where {Dist,Naive} = "glmnet $(Dist.name.name) $(Naive ? "naive" : "cov")"
Base.string(p::LassoBenchmark{LassoOp{Dist,Naive}}) where {Dist,Naive} = "Lasso.jl $(Dist.name.name) $(Naive ? "naive" : "cov")"

inputs = Dict{(Float64, Int, Int),(Matrix{Float64},Vector{Float64})}()
cfgs = vec([begin
                x = (ρ, N, p)
                inputs[x] = makeXY(ρ, N, p)
                x
            end for ρ in [0, 0.1, 0.2, 0.5, 0.9, 0.95],
               (N, p) in [(1000, 100), (5000, 100), (100, 200), (100, 500), (100, 1000), (100, 5000), #=(100, 20000)=#]])
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
