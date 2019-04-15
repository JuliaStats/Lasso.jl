using MLBase, Random

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
@test segminAICc == 71
@test coefsAICc == β[:,segminAICc]

coefsBIC = coef(path, MinBIC())
segminBIC = Lasso.minBIC(path)
@test segminBIC == 55
@test coefsBIC == β[:,segminBIC]

coefsAIC = coef(path, MinAIC())
segminAIC = Lasso.minAIC(path)
@test segminAIC == 71
@test coefsAIC == β[:,segminAIC]

Random.seed!(13)
gen = Kfold(length(y),10)
segCVmin = cross_validate_path(path,MinCVmse(gen))
coefsCVmin = coef(path, MinCVmse(path))
@test segCVmin == 71
@test coefsCVmin == β[:,segCVmin]

Random.seed!(13)
gen = Kfold(length(y),10)
segCVmin = cross_validate_path(path,X,y, MinCVmse(gen), offset=offset)
coefsCVmin = coef(path, MinCVmse(path))
@test segCVmin == 71
@test coefsCVmin == β[:,segCVmin]

Random.seed!(13)
coefsCV1se = coef(path, MinCV1se(path, 20))
Random.seed!(13)
segCV1se = cross_validate_path(path,X,y, MinCV1se(path, 20),offset=offset)
@test segCV1se == 42
@test coefsCV1se == β[:,segCV1se]

end
