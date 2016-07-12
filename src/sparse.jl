"Sparse vector of weights, where non-explictly specified values are assigned the default value (one)"
immutable SparseWeights{T}
     default::T               # default entry represented by no explicit value, 1 if not specified
     spvec::SparseMatrixCSC   # underlying zero-based sparse vector

     SparseWeights(m::Int;default::T=one(T)) = new(default,spzeros(T,m,1))
end

import Base.getindex, Base.setindex!, Base.convert, Base.length, Base.sum, Base.endof, Base.*, Base.show
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

function Base.show(io::IO, w::SparseWeights)
    S = w.spvec
    print(io, S.m, "×", S.n, " sparse weigths vector with ", nnz(S), " ", eltype(S), " entries different from the default ($(w.default))")
    # following code is from SparseMatrixCSC's show
    if nnz(S) == 0
        print(io,".")
        return nothing
    else
        print(io,":")
    end
    limit = false #::Bool = get(io, :limit, false) # did not work in 0.4.5
    if limit
        rows = displaysize(io)[1]
        half_screen_rows = div(rows - 8, 2)
    else
        half_screen_rows = typemax(Int)
    end
    pad = ndigits(max(S.m,S.n))
    k = 0
    sep = "\n\t"
    # io = IOContext(io)
    # if !haskey(io, :compact)
    #     io = IOContext(io, :compact => true)
    # end
    for col = 1:S.n, k = S.colptr[col] : (S.colptr[col+1]-1)
        if k < half_screen_rows || k > nnz(S)-half_screen_rows
            print(io, sep, '[', rpad(S.rowval[k], pad), "]  =  ")
            if isassigned(S.nzval, k)
                # add w.default to entries
                Base.show(io, S.nzval[k]+w.default)
            else
                print(io, Base.undef_ref_str)
            end
        elseif k == half_screen_rows
            print(io, sep, '\u22ee')
        end
        k += 1
    end
end
# rescales A so that it sums to base
rescale(A,base) = A * (base / sum(A))
