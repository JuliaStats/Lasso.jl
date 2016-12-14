using MLBase

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
offset = fill(0.001,size(y))

@time glp = fit(GammaLassoPath, X, y, dist, link, λ=λ,γ=γ,standardize=true, λminratio=0.001, offset=offset)
path = glp

plot(path)

@time coefsAICc = coef(path;select=:AICc)
srand(13)
@time coefsCVmin = coef(path;select=:CVmin)
srand(13)
@time coefsCV1se = coef(path;select=:CV1se,nCVfolds=100)
# fieldnames(path.m.pp)
# y == path.m.rr.y
# offset == path.m.rr.offset
# path.m.pp.X
#
# size(path.m.pp.X)
# size(convert(path.Xnorm)

# gen = LOOCV(nobs(path))
# T = eltype(λ)
# offset=Array(T,0)
ix=1:length(y)

# plot(path)

srand(13); gen = Kfold(length(y[ix]),10)
@time segCVmin = cross_validate_path(path;gen=gen)

srand(13); gen = Kfold(length(y[ix]),10)
@time segCVmin = cross_validate_path(path,X[ix,:],y[ix];offset=offset[ix],gen=gen)

srand(13); gen = Kfold(length(y[ix]),10)
@time segCV1se = cross_validate_path(path,X[ix,:],y[ix];select=:CV1se,gen=gen,offset=offset[ix])

λCVmin = path.λ[segCVmin]
λCV1se = path.λ[segCV1se]
