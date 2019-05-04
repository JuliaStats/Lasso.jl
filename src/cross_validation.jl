using MLBase
# from Taddy (2015, distrom):
# CV selection rules: both CV1se, which chooses the largest λt with mean
# OOS deviance no more than one standard error away from minimum, and CVmin,
# which chooses λt at lowest mean OOS deviance.

function CVmin(oosdevs)
    cvmeans = vec(mean(oosdevs,dims=2))
    segCVmin = argmin(cvmeans)
end

function CV1se(oosdevs)
    nλ,nfolds = size(oosdevs)
    cvmeans = vec(mean(oosdevs,dims=2))
    (mincvmean,segCVmin) = findmin(cvmeans)
    mincvstds = std(view(oosdevs, segCVmin, :)) / sqrt(nfolds-1)
    mincvmean_plus_mincvstds = mincvmean + mincvstds
    for s=1:nλ
        cv1se = mincvmean_plus_mincvstds - cvmeans[s]
        if cv1se >= 0
            return s
        end
    end
    error("should have found the cv1se by now")
end

pathtype(::LassoPath) = LassoPath
pathtype(::GammaLassoPath) = GammaLassoPath

function cross_validate_path(path::R,    # fitted path
                       X::AbstractMatrix{T}, y::V,        # potentially new data
                       select::S;
                       offset::FPVector=T[],
                       fitargs...) where {R<:RegularizationPath,S<:CVSegSelect,T<:AbstractFloat,V<:FPVector}
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

    # always predict with all path segments
    allseg = AllSeg()

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
        μ = predict(foldpath, X[test_inds,:]; offset=foldoffset, select=allseg)

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
