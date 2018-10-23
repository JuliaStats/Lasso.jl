module TrendFiltering
using LinearAlgebra, SparseArrays, StatsBase, ..FusedLassoMod, ..Util
using DSP: filt!
import Base: +, -, *
export TrendFilter

# Implements the algorithm from Ramdas, A., & Tibshirani, R. J. (2014).
# Fast and flexible ADMM algorithms for trend filtering. arXiv
# Preprint arXiv:1406.2082. Retrieved from
# http://arxiv.org/abs/1406.2082

struct DifferenceMatrix{T} <: AbstractMatrix{T}
    k::Int
    n::Int
    b::Vector{T}                  # Coefficients for mul!
    si::Vector{T}                 # State for mul!/At_mul_B!

    function DifferenceMatrix{T}(k, n) where T
        n >= 2*k+2 || throw(ArgumentError("signal must have length >= 2*order+2"))
        b = T[ifelse(isodd(i), -1, 1)*binomial(k+1, i) for i = 0:k+1]
        new(k, n, b, zeros(T, k+1))
    end
end

Base.size(K::DifferenceMatrix) = (K.n-K.k-1, K.n)

# Multiply by difference matrix by filtering
function LinearAlgebra.mul!(out::AbstractVector, K::DifferenceMatrix, x::AbstractVector, α::Real=1)
    length(x) == size(K, 2) || throw(DimensionMismatch("length(x) == $(length(x)) != $(size(K, 2)) == size(K, 2)"))
    length(out) == size(K, 1) || throw(DimensionMismatch("length(x) == $(length(x)) != $(size(K, 1)) == size(K, 1)"))
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
*(K::DifferenceMatrix, x::AbstractVector) = mul!(similar(x, size(K, 1)), K, x)

function LinearAlgebra.mul!(out::AbstractVector, K::Adjoint{<:Any,<:DifferenceMatrix}, x::AbstractVector, α::Real=1)
    length(x) == size(K.parent, 1) || throw(DimensionMismatch("length(x) == $(length(x)) != $(size(K.parent, 1)) == size(K.parent, 1)"))
    length(out) == size(K.parent, 2) || throw(DimensionMismatch("length(out) == $(length(out)) != $(size(K.parent, 2)) == size(K.parent, 2)"))
    b = K.parent.b
    si = fill!(K.parent.si, 0)
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
LinearAlgebra.mul!(out::AbstractVector, K::Hermitian{<:Any,<:DifferenceMatrix}, x::AbstractVector, α::Real=1) = mul!(out, K.parent', x)
*(K::Adjoint{<:Any,<:DifferenceMatrix}, x::AbstractVector) = mul!(similar(x, size(K.parent, 2)), K, x)
*(K::Hermitian{<:Any,<:DifferenceMatrix}, x::AbstractVector) = mul!(similar(x, size(K.parent, 2)), K.parent', x)

# Product with self, efficiently
function *(K::Adjoint{<:Any,<:DifferenceMatrix}, K2::DifferenceMatrix)
    K.parent === K2 || error("matrix multiplication only supported with same difference matrix")
    computeDtD(K.parent.b, K.parent.n)
end
*(K::Hermitian{<:Any,<:DifferenceMatrix}, K2::DifferenceMatrix) = *(K.parent, K2::DifferenceMatrix)

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
    colptr = Vector{Int}(undef, n+1)
    rowval = Vector{Int}(undef, (k+2)*(n-k-1)+(k+1)*n)
    nzval = Vector{Float64}(undef, (k+2)*(n-k-1)+(k+1)*n)
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

mutable struct TrendFilter{T,S,VT}
    Dkp1::DifferenceMatrix{T}                # D(k+1)
    Dk::DifferenceMatrix{T}                  # D(k)
    DktDk::SparseMatrixCSC{T,Int}            # Dk'Dk
    β::Vector{T}                             # Output coefficients and temporary storage for ρD(k+1)'α + u
    u::Vector{T}                             # ADMM u
    Dkβ::Vector{T}                           # Temporary storage for D(k)*β
    Dkp1β::VT                                # Temporary storage for D(k+1)*β (aliases Dkβ)
    flsa::FusedLasso{T,S}                    # Fused lasso model
    niter::Int                               # Number of ADMM iterations
end

function StatsBase.fit(::Type{TrendFilter}, y::AbstractVector{T}, order, λ; dofit::Bool=true, args...) where T
    order >= 1 || throw(ArgumentError("order must be >= 1"))
    Dkp1 = DifferenceMatrix{T}(order, length(y))
    Dk = DifferenceMatrix{T}(order-1, length(y))
    β = zeros(T, length(y))
    u = zeros(T, size(Dk, 1))
    Dkβ = zeros(T, size(Dk, 1))
    Dkp1β = view(Dkβ, 1:size(Dkp1, 1))
    tf = TrendFilter(Dkp1, Dk, Dk'Dk, β, u, Dkβ, Dkp1β, fit(FusedLasso, Dkβ, λ; dofit=false), -1)
    dofit && fit!(tf, y, λ; args...)
    return tf
end

function StatsBase.fit!(tf::TrendFilter{T}, y::AbstractVector{T}, λ::Real; niter=100000, tol=1e-6, ρ=λ) where T
    @extractfields tf Dkp1 Dk DktDk β u Dkβ Dkp1β flsa
    length(y) == length(β) || throw(ArgumentError("input size $(length(y)) does not match model size $(length(β))"))

    # Reuse this memory
    ρDtαu = β
    αpu = Dkβ

    fact = cholesky(sparse(1.0I, size(Dk, 2), size(Dk, 2)) + ρ*DktDk)
    α = coef(flsa)
    fill!(α, 0)

    oldobj = obj = Inf
    local iter
    for outer iter = 1:niter
        # Eq. 11 (update β)
        broadcast!(+, αpu, α, u)
        mul!(ρDtαu, Dk', αpu, ρ)
        β = fact\broadcast!(+, ρDtαu, ρDtαu, y)

        # Check for convergence
        mul!(Dkp1β, Dkp1, β)
        oldobj = obj
        obj = sumsqdiff(y, β)/2 + λ*sum(abs, Dkp1β)
        abs(oldobj - obj) < abs(obj * tol) && break

        # Eq. 12 (update α)
        mul!(Dkβ, Dk, β)
        broadcast!(-, u, Dkβ, u)
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
