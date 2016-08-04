using MLBase

function cross_validate_path{T<:AbstractFloat,V<:FPVector}(path::RegularizationPath,    # fitted path
                                                     X::AbstractMatrix{T}, y::V,        # potentially new data
                                                     gen=Kfold(nobs(path),10);          # folds generator (see MLBase)
                                                     wts::FPVector=ones(T, length(y)), # observation weights
                                                     fitargs...)
    λ = path.λ
    n = nobs(path)
    nfolds = length(gen)
    nλ = length(λ)
    d = distfun(path)
    l = linkfun(path)
    oosdevs = zeros(T,nλ,nfolds)
    #TODO: find a more elegant way of identifying pathType
    if typeof(path) <: LassoPath
     pathType = LassoPath
    elseif typeof(path) <: GammaLassoPath
     pathType = GammaLassoPath
    else
     error("unknown path typeof(path) $(typeof(path))")
    end
    # temp storage for devs
    devresidv = zeros(T,n,nλ)
    for (j, train_inds) in enumerate(gen)
     test_inds = setdiff(1:n, train_inds)
     foldpath = fit(pathType,X[train_inds,:],y[train_inds],d,l;λ=λ,fitargs...)
     oosdevs[:,j] = deviance!(devresidv,foldpath,X[test_inds,:],y[test_inds])
    end
    oosdevs
end

datapath = joinpath(dirname(@__FILE__), "..","test","data")

(family, dist, link) = (("gaussian", Normal(), IdentityLink()), ("binomial", Binomial(), LogitLink()), ("poisson", Poisson(), LogLink()))[1]
data = readcsv(joinpath(datapath,"gamlr.$family.data.csv"))
y = data[:,1]
X = data[:,2:end]
(n,p) = size(X)
γ = [0 2 10][1]
fitname = "gamma$γ"
# get gamlr params and estimates
params = readtable(joinpath(datapath,"gamlr.$family.$fitname.params.csv"))
fittable = readtable(joinpath(datapath,"gamlr.$family.$fitname.fit.csv"))
gcoefs = convert(Matrix{Float64},readcsv(joinpath(datapath,"gamlr.$family.$fitname.coefs.csv")))
family = params[1,:fit_family]
λ = nothing #convert(Vector{Float64},fittable[:fit_lambda]) # should be set to nothing evenatually
# fit julia version
@time glp = fit(GammaLassoPath, X, y, dist, link, λ=λ,γ=γ,standardize=true, λminratio=0.001)

path = glp
gen = Kfold(nobs(path),7)
fitargs = ()
T = eltype(λ)
@time oosdevs = cross_validate_path(path,X,y)

λ = path.λ
n = nobs(path)
nfolds = length(gen)
nλ = length(λ)
d = distfun(path)
l = linkfun(path)
oosdevs = zeros(T,nλ,nfolds)
#TODO: find a more elegant way of identifying pathType
if typeof(path) <: LassoPath
 pathType = LassoPath
elseif typeof(path) <: GammaLassoPath
 pathType = GammaLassoPath
else
 error("unknown path typeof(path) $(typeof(path))")
end
# temp storage for devs
devresidv = zeros(T,n,nλ)
for (j, train_inds) in enumerate(gen)
 test_inds = setdiff(1:n, train_inds)
 foldpath = fit(pathType,X[train_inds,:],y[train_inds],d,l;λ=λ,fitargs...)
 oosdevs[:,j] = deviance!(devresidv,foldpath,X[test_inds,:],y[test_inds])
end
oosdevs

using ProfileView
Profile.init(delay=0.001)
Profile.clear()
@profile oosdevs = cross_validate_path(path,X,y);
ProfileView.view()
# Profile.print()
