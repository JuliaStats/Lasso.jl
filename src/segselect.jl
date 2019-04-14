"RegularizationPath segment selector supertype"
abstract type SegSelect end

"A RegularizationPath segment selector that returns all segments"
struct AllSeg <: SegSelect end

"Selects the RegularizationPath segment with the minimum AIC"
struct MinAIC <: SegSelect end

"Index of the selected RegularizationPath segment"
segselect(path::RegularizationPath, select::MinAIC) = minAIC(path)

"Selects the RegularizationPath segment with the minimum corrected AIC"
struct MinAICc <: SegSelect
    k::Int # k parameter used to correct AIC criterion
    MinAICc(k::Int=2) = new(k)
end
segselect(path::RegularizationPath, select::MinAICc) = minAICc(path; k=select.k)

"Selects the RegularizationPath segment with the minimum BIC"
struct MinBIC <: SegSelect end

segselect(path::RegularizationPath, select::MinBIC) = minBIC(path)

"RegularizationPath segment selector supertype"
abstract type CVSegSelect <: SegSelect end

"Selects the RegularizationPath segment with the minimum cross-validation mse"
struct MinCVmse <: CVSegSelect
    gen::CrossValGenerator

    MinCVmse(gen::CrossValGenerator) = new(gen)
    MinCVmse(path::RegularizationPath, k::Int=10) = new(Kfold(length(path.m.rr.y), k))
end

CVfun(oosdevs, ::MinCVmse) = CVmin(oosdevs)

"""
Selects the RegularizationPath segment with the largest λt with mean
OOS deviance no more than one standard error away from minimum
"""
struct MinCV1se <: CVSegSelect
    gen::CrossValGenerator

    MinCV1se(gen::CrossValGenerator) = new(gen)
    MinCV1se(path::RegularizationPath, k::Int=10) = new(Kfold(length(path.m.rr.y), k))
end

CVfun(oosdevs, ::MinCV1se) = CV1se(oosdevs)

StatsBase.coef(path::RegularizationPath; select::S=AllSeg()) where S <: SegSelect = coef(path, select)
function StatsBase.coef(path::RegularizationPath, select::S) where S <: SegSelect
    if !isdefined(path,:coefs)
        X = path.m.pp.X
        p,nλ = size(path)
        return zeros(eltype(X),p)
    end

    seg = segselect(path, select)

    if hasintercept(path)
        vec(vcat(path.b0[seg],path.coefs[:,seg]))
    else
        path.coefs[:,seg]
    end
end

function StatsBase.coef(path::RegularizationPath, select::AllSeg)
    if !isdefined(path,:coefs)
        X = path.m.pp.X
        p,nλ = size(path)
        return spzeros(eltype(X),p,nλ)
    end

    if hasintercept(path)
        vcat(path.b0',path.coefs)
    else
        path.coefs
    end
end

function cross_validate_path(path::RegularizationPath,    # fitted path
                       X::AbstractMatrix{T}, y::V,        # potentially new data
                       select::S;
                       offset::FPVector=T[],
                       fitargs...) where {T<:AbstractFloat,V<:FPVector, S<:CVSegSelect}
    @extractfields path m λ
    gen = select.gen
    n,p = size(X)
    @assert n == length(y) "size(X,1) != length(y)"

    nfolds = length(gen)
    nλ = length(λ)
    d = distfun(path)
    l = linkfun(path)

    # valid offset given?
    if length(m.rr.offset) > 0
        length(offset) == n ||
            throw(ArgumentError("fit with offset, so `offset` kw arg must be an offset of length $n"))
    else
        length(offset) > 0 && throw(ArgumentError("fit without offset, so value of `offset` kw arg does not make sense"))
    end

    # EQUAL WEIGHTS ONLY!
    wts = ones(T, n)

    # results array
    oosdevs = zeros(T,nλ,nfolds)

    for (f, train_inds) in enumerate(gen)
        test_inds = setdiff(1:n, train_inds)
        nis = length(test_inds)

        if length(offset) > 0
            foldoffset = offset[train_inds]
        else
            foldoffset = offset
        end

        # fit model to train_inds
        foldpath = fit(pathtype(path),X[train_inds,:],y[train_inds],d,l;λ=λ,wts=wts[train_inds],offset=foldoffset,fitargs...)

        if length(offset) > 0
            foldoffset = offset[test_inds]
        else
            foldoffset = offset
        end

        # calculate etas for each obs x segment
        μ = predict(foldpath, X[test_inds,:]; offset=foldoffset, select=select)

        # calculate deviations on test sets efficiently (not much mem)
        for s=1:nλ
            # deviance of segment s (cummulator for sum of obs deviances)
            devs = zero(T)

            for ip=1:nis
                # get test obs
                yi = y[test_inds[ip]]

                # deviance of a single observation i in segment s
                devs += devresid(d, yi, μ[ip,s])
            end

            # store result
            oosdevs[s,f] = devs/nis
        end
    end

    CVfun(oosdevs, select)
end

# convenience function to use the same data as in original path
function cross_validate_path(path::RegularizationPath,     # fitted path
                        select::S;
                        fitargs...) where S<:CVSegSelect
    m = path.m
    y = m.rr.y
    offset = m.rr.offset
    Xstandardized = m.pp.X
    cross_validate_path(path,Xstandardized,y,select;
        offset=offset,standardize=false,fitargs...)
end

segselect(path::RegularizationPath, select::S) where S<:CVSegSelect =
    cross_validate_path(path, select)

segselect(path::RegularizationPath,
           X::AbstractMatrix{T}, y::V,        # potentially new data
           select::S;
           kwargs...) where {T<:AbstractFloat,V<:FPVector, S<:CVSegSelect} =
    cross_validate_path(path, X, y, select; kwargs...)
