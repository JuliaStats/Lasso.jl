"Sparse vector of weights, where non-explictly specified values are assigned the default value (one)"
immutable SparseWeights{T}
     default::T               # default entry represented by no explicit value, 1 if not specified
     spvec::SparseMatrixCSC   # underlying zero-based sparse vector

     SparseWeights(m::Int;default::T=one(T)) = new(default,spzeros(T,m,1))
end

import Base.getindex, Base.setindex!, Base.convert, Base.length, Base.sum, Base.endof, Base.*
getindex(A::SparseWeights, I) = getindex(A.spvec,I) + A.default
function getindex(A::SparseWeights, i::Int)
  x = getindex(A.spvec,i)
  if x == 0
    A.default
  else
    x + A.default
  end
end
setindex!(A::SparseWeights, v, I) = setindex!(A.spvec, v - A.default, I)
isalldefault(A::SparseWeights) = nnz(A.spvec) == 0
length(A::SparseWeights) = length(A.spvec)
function convert{T}(::Type{SparseWeights},x::AbstractVector{T}; default=one(T))
     w = SparseWeights{T}(length(x); default=default)
     for i=1:length(x)
          if x[i] != default
               w[i] = x[i]
          end
     end
     w
end
convert{T}(::Type{Vector},w::SparseWeights{T}) = convert(Vector,vec(w[1:length(w)]))
endof(A::SparseWeights) = endof(A.spvec)
sum(A::SparseWeights) = sum(A.spvec) + A.default * length(A)
function *{T}(A::SparseWeights{T},x::Real)
     B = SparseWeights{T}(length(A); default=A.default*x)
     for i=eachindex(A.spvec)
          B.spvec[i] = x * A.spvec[i]
     end
     B
end

# rescales A so that it sums to base
rescale(A,base) = A * (base / sum(A))
