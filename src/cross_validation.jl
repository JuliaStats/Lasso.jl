using MLBase

# from Taddy (2015, distrom):
# CV selection rules: both CV1se, which chooses the largest λt with mean
# OOS deviance no more than one standard error away from minimum, and CVmin,
# which chooses λt at lowest mean OOS deviance.

function CVmin(oosdevs)
  cvmeans = mean(oosdevs,2)
  segCVmin = indmin(cvmeans)
end

# function CV1se_slow_but_sure(oosdevs)
#   cvmeans = vec(mean(oosdevs,2))
#   (mincvmean,segCVmin) = findmin(cvmeans)
#   cvstds = vec(std(oosdevs,2)) / sqrt(nfolds-1)
#   cv1se = (mincvmean + cvstds[segCVmin]) - cvmeans
#   segCV1se = minimum(collect(1:size(cvmeans,1))[cv1se.>=0])
# end

function CV1se(oosdevs)
  nλ,nfolds = size(oosdevs)
  cvmeans = vec(mean(oosdevs,2))
  (mincvmean,segCVmin) = findmin(cvmeans)
  mincvstds = std(oosdevs[segCVmin,:]) / sqrt(nfolds-1)
  mincvmean_plus_mincvstds = mincvmean + mincvstds
  for s=1:nλ
    cv1se = mincvmean_plus_mincvstds - cvmeans[s]
    if cv1se >= 0
      return s
    end
  end
  error("should have found the cv1se by now")
end

function cross_validate_path_slow_but_sure{T<:AbstractFloat,V<:FPVector}(path::RegularizationPath,    # fitted path
                                                     X::AbstractMatrix{T}, y::V;        # potentially new data
                                                     gen=Kfold(length(y),10),           # folds generator (see MLBase)
                                                     select=:CVmin,                     # :CVmin or :CV1se
                                                     fitargs...)
    n,p = size(X)
    @assert n == length(y) "size(X,1) != length(y)"

    nfolds = length(gen)
    λ = path.λ
    nλ = length(λ)
    d = distfun(path)
    l = linkfun(path)

    # allocate space for results
    oosdevs = zeros(T,nλ,nfolds)

    #TODO: find a more elegant way of identifying pathType
    if typeof(path) <: LassoPath
     pathType = LassoPath
    elseif typeof(path) <: GammaLassoPath
     pathType = GammaLassoPath
    else
     error("unknown path typeof(path) $(typeof(path))")
    end

    # EQUAL WEIGHTS ONLY!
    wts = ones(T, n)

    for (j, train_inds) in enumerate(gen)
     test_inds = setdiff(1:n, train_inds)
     if length(offset) > 0
       foldoffset = offset[train_inds]
     else
       foldoffset = offset
     end

     # fit model to train_inds
     foldpath = fit(pathType,X[train_inds,:],y[train_inds],d,l;λ=λ,wts=wts[train_inds],offset=foldoffset,fitargs...)

     if length(offset) > 0
       foldoffset = offset[test_inds]
     else
       foldoffset = offset
     end

     oosdevs[:,j] = deviance(foldpath,X[test_inds,:],y[test_inds];offset=foldoffset)
    end

    CVfun = eval(select)
    CVfun(oosdevs)
end

# convenience function to use the same data as in original path
function cross_validate_path(path::RegularizationPath;                                  # fitted path
                                                     gen=Kfold(length(y),10),           # folds generator (see MLBase)
                                                     select=:CVmin)                     # :CVmin or :CV1se
    m = path.m
    y = m.rr.y
    offset = m.rr.offset
    Xstandardized = m.pp.X
    cross_validate_path(path,Xstandardized,y;gen=gen,select=select,offset=offset,standardize=false)
end

function cross_validate_path{T<:AbstractFloat,V<:FPVector}(path::RegularizationPath,    # fitted path
                                                     X::AbstractMatrix{T}, y::V;        # potentially new data
                                                     gen=Kfold(length(y),10),           # folds generator (see MLBase)
                                                     select=:CVmin,                     # :CVmin or :CV1se
                                                     offset::FPVector=Array(T,0),
                                                     fitargs...)
    @extractfields path m λ
    n,p = size(X)
    @assert n == length(y) "size(X,1) != length(y)"

    nfolds = length(gen)
    nλ = length(λ)
    d = distfun(path)
    l = linkfun(path)

    #TODO: find a more elegant way of identifying pathType
    if typeof(path) <: LassoPath
     pathType = LassoPath
    elseif typeof(path) <: GammaLassoPath
     pathType = GammaLassoPath
    else
     error("unknown path typeof(path) $(typeof(path))")
    end

    # valid offset given?
    if length(m.rr.offset) > 0
        length(offset) == n ||
            throw(ArgumentError("fit with offset, so `offset` kw arg must be an offset of length $n"))
    else
        length(offset) > 0 && throw(ArgumentError("fit without offset, so value of `offset` kw arg does not make sense"))
    end

    # EQUAL WEIGHTS ONLY!
    wts = ones(T, n)
    devresid_closure(yi, μis) = devresid(d, yi, μis, one(T))

    # add an interecept to X if the model has one
    if hasintercept(path)
      Xwconst = [ones(T,n,1) X]
    else
      Xwconst = X
    end

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
      foldpath = fit(pathType,X[train_inds,:],y[train_inds],d,l;λ=λ,wts=wts[train_inds],offset=foldoffset,fitargs...)

      if length(offset) > 0
        foldoffset = offset[test_inds]
      else
        foldoffset = offset
      end

      # calculate etas for each obs x segment
      μ = predict(foldpath, X[test_inds,:]; offset=foldoffset, select=:all)

      # calculate deviations on test sets efficiently (not much mem)
      for s=1:nλ
        # deviance of segment s (cummulator for sum of obs deviances)
        devs = zero(T)

        @inbounds @simd for ip=1:nis
          # get test obs
          yi = y[test_inds[ip]]

          # deviance of a single observation i in segment s
          devs += devresid_closure(yi, μ[ip,s])
        end

        # store result
        oosdevs[s,f] = devs/nis
      end
    end

    CVfun = eval(select)
    CVfun(oosdevs)
end
