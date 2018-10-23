module FusedLassoMod
using StatsBase
import Base: +, -, *
export FusedLasso

# Implements the algorithm described in Johnson, N. A. (2013). A
# Dynamic Programming Algorithm for the Fused Lasso and
# L0-Segmentation. Journal of Computational and Graphical Statistics,
# 22(2), 246–260. doi:10.1080/10618600.2012.681238

struct NormalCoefs{T}
    lin::T
    quad::T

    NormalCoefs{T}(lin::Real) where {T} = new(lin, 0)
    NormalCoefs{T}(lin::Real, quad::Real) where {T} = new(lin, quad)
end
+(a::NormalCoefs{T}, b::NormalCoefs{T}) where {T} = NormalCoefs{T}(a.lin+b.lin, a.quad+b.quad)
-(a::NormalCoefs{T}, b::NormalCoefs{T}) where {T} = NormalCoefs{T}(a.lin-b.lin, a.quad-b.quad)
+(a::NormalCoefs{T}, b::Real) where {T} = NormalCoefs{T}(a.lin+b, a.quad)
-(a::NormalCoefs{T}, b::Real) where {T} = NormalCoefs{T}(a.lin-b, a.quad)
*(a::Real, b::NormalCoefs{T}) where {T} = NormalCoefs{T}(a*b.lin, a*b.quad)

# Implements Algorithm 2 lines 8 and 19
solveforbtilde(a::NormalCoefs{T}, lhs::Real) where {T} = (lhs - a.lin)/(2 * a.quad)

# These are marginally faster than computing btilde explicitly because
# they avoid division
btilde_lt(a::NormalCoefs{T}, lhs::Real, x::Real) where {T} = lhs - a.lin > 2 * a.quad * x
btilde_gt(a::NormalCoefs{T}, lhs::Real, x::Real) where {T} = lhs - a.lin < 2 * a.quad * x

struct Knot{T,S}
    pos::T
    coefs::S
    sign::Int8
end

struct FusedLasso{T,S} <: RegressionModel
    β::Vector{T}              # Coefficients
    knots::Vector{Knot{T,S}}  # Active knots
    bp::Matrix{T}             # Backpointers
end

function StatsBase.fit(::Type{FusedLasso}, y::AbstractVector{T}, λ::Real; dofit::Bool=true) where T
    S = NormalCoefs{T}
    flsa = FusedLasso{T,S}(Array{T}(undef, length(y)), Array{Knot{T,S}}(undef, 2), Array{T}(undef, 2, length(y)-1))
    dofit && fit!(flsa, y, λ)
    flsa
end

function StatsBase.fit!(flsa::FusedLasso{T,S}, y::AbstractVector{T}, λ::Real) where {T,S}
    β = flsa.β
    knots = flsa.knots
    bp = flsa.bp

    length(y) == length(β) || throw(ArgumentError("input size $(length(y)) does not match model size $(length(β))"))

    resize!(knots, 2)
    knots[1] = Knot{T,S}(-Inf, S(0), 1)
    knots[2] = Knot{T,S}(Inf, S(0), -1)

    # Algorithm 1 lines 2-5
    @inbounds for k = 1:length(y)-1
        t1 = 0
        t2 = 0
        aminus = NormalCoefs{T}(y[k], -0.5)                # Algorithm 2 line 4
        for outer t1 = 1:length(knots)-1                         # Algorithm 2 line 5
            knot = knots[t1]
            aminus += knot.sign*knot.coefs                 # Algorithm 2 line 6
            btilde_lt(aminus, λ, knots[t1+1].pos) && break # Algorithm 2 line 7-8
        end
        bminus = solveforbtilde(aminus, λ)

        aplus = NormalCoefs{T}(y[k], -0.5)                 # Algorithm 2 line 15
        t2 = length(knots)
        while t2 >= 2                                      # Algorithm 2 line 16
            knot = knots[t2]
            aplus -= knot.sign*knot.coefs                  # Algorithm 2 line 17
            btilde_gt(aplus, -λ, knots[t2-1].pos) && break # Algorithm 2 line 18-19
            t2 -= 1
        end
        bplus = solveforbtilde(aplus, -λ)

        # Resize knots so that we have only knots[t1+1:t2-1] and 2
        # elements at either end. It would be better to use a different
        # data structure here.
        estlen = t2 - t1 + 3
        if estlen == 4
            resize!(knots, 4)
        else
            if t2 == length(knots)
                resize!(knots, t2+1)
            else
                deleteat!(knots, t2+2:length(knots))
            end
            if t1 == 1
                pushfirst!(knots, Knot{T,S}(-Inf, S(0), 1))
            else
                deleteat!(knots, 1:t1-2)
            end
        end
        knots[1] = Knot{T,S}(-Inf, S(λ), 1)                # Algorithm 2 line 28
        knots[2] = Knot{T,S}(bminus, aminus-λ, 1)          # Algorithm 2 line 29
        knots[end-1] = Knot{T,S}(bplus, aplus+λ, -1)       # Algorithm 2 line 20
        knots[end] = Knot{T,S}(Inf, S(-λ), -1)             # Algorithm 2 line 31
        bp[1, k] = bminus
        bp[2, k] = bplus
    end

    # Algorithm 1 line 6
    aminus = NormalCoefs{T}(y[end], -0.5)
    for t1 = 1:length(knots)
        knot = knots[t1]
        aminus += knot.sign*knot.coefs
        btilde_lt(aminus, 0, knots[t1+1].pos) && break
    end
    β[end] = solveforbtilde(aminus, 0)

    # Backtrace
    for k = length(y)-1:-1:1                        # Algorithm 1 line 6
        β[k] = min(bp[2, k], max(β[k+1], bp[1, k])) # Algorithm 1 line 7
    end
    flsa
end

StatsBase.coef(flsa::FusedLasso) = flsa.β

end
