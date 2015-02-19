module TrendFiltering
using StatsBase, ..FusedLassoMod, ..Util
export TrendFilter

# Implements the algorithm from Ramdas, A., & Tibshirani, R. J. (2014).
# Fast and flexible ADMM algorithms for trend filtering. arXiv
# Preprint arXiv:1406.2082. Retrieved from
# http://arxiv.org/abs/1406.2082

immutable DifferenceMatrix{T} <: AbstractMatrix{T}
    k::Int
    n::Int
    b::Vector{T}                  # Coefficients for A_mul_B!
    si::Vector{T}                 # State for A_mul_B!/At_mul_B!

    function DifferenceMatrix(k, n)
        b = T[ifelse(isodd(i), -1, 1)*binomial(k+1, i) for i = 0:k+1]
        new(k, n, b, zeros(T, k+1))
    end
end

Base.size(K::DifferenceMatrix) = (K.n-K.k-1, K.n)

# Multiply by difference matrix by filtering
function Base.LinAlg.A_mul_B!(out::AbstractVector, K::DifferenceMatrix, x::AbstractVector, α::Real=1)
    b = K.b
    si = fill!(K.si, 0)
    silen = length(b)-1
    @inbounds for i = 1:length(x)
        xi = x[i]
        val = si[1] + b[1]*xi
        for j=1:(silen-1)
            si[j] = si[j+1] + b[j+1]*xi
        end
        si[silen] = b[silen+1]*xi
        if i > silen
            out[i-silen] = α*val
        end
    end
    out
end
*(K::DifferenceMatrix, x::AbstractVector) = A_mul_B!(similar(x, size(K, 1)), K, x)

function Base.LinAlg.At_mul_B!(out::AbstractVector, K::DifferenceMatrix, x::AbstractVector, α::Real=1)
    b = K.b
    si = fill!(K.si, 0)
    silen = length(b)-1
    isodd(silen) && (α = -α)
    n = length(x)
    @inbounds for i = 1:n
        xi = x[i]
        val = si[1] + b[1]*xi
        for j=1:(silen-1)
            si[j] = si[j+1] + b[j+1]*xi
        end
        si[silen] = b[silen+1]*xi
        out[i] = α*val
    end
    @inbounds for i = 1:length(si)
        out[n+i] = α*si[i]
    end
    out
end
Base.LinAlg.Ac_mul_B!(out::AbstractVector, K::DifferenceMatrix, x::AbstractVector, α::Real=1) = At_mul_B!(out, K, x)
Base.LinAlg.At_mul_B(K::DifferenceMatrix, x::AbstractVector) = At_mul_B!(similar(x, size(K, 2)), K, x)
Base.LinAlg.Ac_mul_B(K::DifferenceMatrix, x::AbstractVector) = At_mul_B!(similar(x, size(K, 2)), K, x)

# Product with self, efficiently
function Base.LinAlg.At_mul_B(K::DifferenceMatrix, K2::DifferenceMatrix)
    K === K2 || error("matrix multiplication only supported with same difference matrix")
    computeDtD(K.b, K.n)
end
Base.LinAlg.Ac_mul_B(K::DifferenceMatrix, K2::DifferenceMatrix) = At_mul_B(K::DifferenceMatrix, K2::DifferenceMatrix)

function computeDtD(c, n)
    k = length(c) - 2
    sgn = iseven(k)
    cc = zeros(eltype(c), 2*length(c)-1)
    for i = 1:length(c)
        cc[i] = sgn ? -c[i] : c[i]
    end
    filt!(cc, c, [one(eltype(c))], cc)
    sides = zeros(eltype(c), 2*length(c)-2, length(c)-1)
    for j = 1:length(c)-1
        for i = 1:j
            sides[i, j] = sgn ? -c[i] : c[i]
        end
    end
    filt!(sides, c, [one(eltype(c))], sides)
    colptr = Array(Int, n+1)
    rowval = Array(Int, (k+2)*(n-k-1)+(k+1)*n)
    nzval = Array(Float64, (k+2)*(n-k-1)+(k+1)*n)
    idx = 1
    for i = 1:k+1
        colptr[i] = idx
        for j = 1:k+i+1
            rowval[idx+j-1] = j
            nzval[idx+j-1] = sides[k+2+i-j, i]
        end
        idx += k+i+1
    end
    for i = k+2:n-(k+1)
        colptr[i] = idx
        for j = 1:length(cc)
            rowval[idx+j-1] = i-k+j-2
            nzval[idx+j-1] = cc[j]
        end
        idx += length(cc)
    end
    for i = k+1:-1:1
        colptr[n-i+1] = idx
        for j = 1:i+k+1
            rowval[idx+j-1] = n-k-1-i+j
            nzval[idx+j-1] = sides[j, i]
        end
        idx += i+k+1
    end
    colptr[end] = idx
    return SparseMatrixCSC(n, n, colptr, rowval, nzval)
end

# Sum of squared differences between two vectors
function sumsqdiff(x, y)
    length(x) == length(y) || throw(DimensionMismatch())
    v = zero(Base.promote_eltype(x, y))
    @simd for i = 1:length(x)
        @inbounds v += abs2(x[i] - y[i])
    end
    v
end

# Soft threshold
S(z, γ) = abs(z) <= γ ? zero(z) : ifelse(z > 0, z - γ, z + γ)

type TrendFilter{T,S}
    Dkp1::DifferenceMatrix{T}                # D(k+1)
    Dk::DifferenceMatrix{T}                  # D(k)
    DktDk::SparseMatrixCSC{T,Int}            # Dk'Dk
    β::Vector{T}                             # Output coefficients and temporary storage for ρD(k+1)'α + u
    u::Vector{T}                             # ADMM u
    Dβ::Vector{T}                            # Temporary storage for D(k)*β
    flsa::FusedLasso{T,S}                    # Fused lasso model
    niter::Int                               # Number of ADMM iterations
end

function StatsBase.fit{T}(::Type{TrendFilter}, y::AbstractVector{T}, order, λ; dofit::Bool=true, args...)
    Dkp1 = DifferenceMatrix{T}(order, length(y))
    Dk = DifferenceMatrix{T}(order-1, length(y))
    β = zeros(T, length(y))
    u = zeros(T, size(Dk, 1))
    Dβ = zeros(T, size(Dk, 1))
    tf = TrendFilter(Dkp1, Dk, Dk'Dk, β, u, Dβ, fit(FusedLasso, Dβ, λ; dofit=false), -1)
    dofit && fit!(tf, y, λ; args...)
    return tf
end

function StatsBase.fit!{T}(tf::TrendFilter{T}, y::AbstractVector{T}, λ::Real; niter=100000, tol=1e-6, ρ=λ)
    @extractfields tf Dkp1 Dk DktDk β u Dβ flsa
    length(y) == length(β) || throw(ArgumentError("input size $(length(y)) does not match model size $(length(β))"))

    # Reuse this memory
    ρDtαu = β
    αpu = Dβ

    fact = cholfact(speye(size(Dk, 2)) + ρ*DktDk)
    α = coef(flsa)
    fill!(α, 0)

    oldobj = obj = Inf
    local iter
    for iter = 1:niter
        # Eq. 11 (update β)
        broadcast!(+, αpu, α, u)
        At_mul_B!(ρDtαu, Dk, αpu, ρ)
        β = fact\broadcast!(+, ρDtαu, ρDtαu, y)

        # Check for convergence
        A_mul_B!(Dβ, Dkp1, β)
        oldobj = obj
        obj = sumsqdiff(y, β)/2 + λ*sumabs(Dβ)
        abs(oldobj - obj) < abs(obj * tol) && break

        # Eq. 12 (update α)
        A_mul_B!(Dβ, Dk, β)
        broadcast!(-, u, Dβ, u)
        fit!(flsa, u, λ/ρ)

        # Eq. 13 (update u; u actually contains Dβ - u)
        broadcast!(-, u, α, u)
    end
    if abs(oldobj - obj) > abs(obj * tol)
        error("ADMM did not converge in $niter iterations")
    end

    # Save coefficients
    tf.β = β
    tf.niter = iter
    tf
end

StatsBase.coef(tf::TrendFilter) = tf.β

end