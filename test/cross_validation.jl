using MLBase, Random, StableRNGs

@testset "cross validation" begin

datapath = joinpath(dirname(@__FILE__), "data")

data = readcsvmat(joinpath(datapath,"gamlr.gaussian.data.csv"))
y = data[:,1]
X = data[:,2:end]
offset = fill(0.001,size(y))

path = fit(LassoPath, X, y; offset=offset)
β = coef(path, AllSeg())

coefsAICc = coef(path, MinAICc())
segminAICc = minAICc(path)
@test segminAICc == 72
@test coefsAICc == β[:,segminAICc]

coefsBIC = coef(path, MinBIC())
segminBIC = Lasso.minBIC(path)
@test segminBIC == 55
@test coefsBIC == β[:,segminBIC]

coefsAIC = coef(path, MinAIC())
segminAIC = Lasso.minAIC(path)
@test segminAIC == 72
@test coefsAIC == β[:,segminAIC]

gen = Kfold(StableRNG(13), length(y), 10)
segCVmin = cross_validate_path(path, MinCVmse(gen))
coefsCVmin = coef(path, MinCVmse(Kfold(StableRNG(13), length(y), 10)))
@test segCVmin == 72
@test coefsCVmin == β[:,segCVmin]

gen = Kfold(StableRNG(13), length(y), 10)
segCVmin = cross_validate_path(path, X, y, MinCVmse(gen), offset=offset)
coefsCVmin = coef(path, MinCVmse(Kfold(StableRNG(13), length(y), 10)))
@test segCVmin == 72
@test coefsCVmin == β[:,segCVmin]

gen1 = Kfold(StableRNG(13), length(y), 20)
coefsCV1se = coef(path, MinCV1se(gen1))
gen2 = Kfold(StableRNG(13), length(y), 20)
segCV1se = cross_validate_path(path, X, y, MinCV1se(gen2), offset=offset)
@test segCV1se == 41
@test coefsCV1se == β[:,segCV1se]

end
