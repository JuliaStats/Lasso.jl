using MLBase, Random

@testset "cross validation" begin

datapath = joinpath(dirname(@__FILE__), "data")

data = readcsvmat(joinpath(datapath,"gamlr.gaussian.data.csv"))
y = data[:,1]
X = data[:,2:end]
offset = fill(0.001,size(y))

path = fit(LassoPath, X, y; offset=offset)
β = coef(path; select=:all)

coefsAICc = coef(path;select=:AICc)
segminAICc = minAICc(path)
@test segminAICc == 71
@test coefsAICc == β[:,segminAICc]

coefsBIC = coef(path;select=:BIC)
segminBIC = Lasso.minBIC(path)
@test segminBIC == 55
@test coefsBIC == β[:,segminBIC]

coefsAIC = coef(path;select=:AIC)
segminAIC = Lasso.minAIC(path)
@test segminAIC == 71
@test coefsAIC == β[:,segminAIC]

Random.seed!(13)
gen = Kfold(length(y),10)
segCVmin = cross_validate_path(path;gen=gen)
coefsCVmin = coef(path;select=:CVmin)
@test segCVmin == 71
@test coefsCVmin == β[:,segCVmin]

Random.seed!(13)
gen = Kfold(length(y),10)
segCVmin = cross_validate_path(path,X,y; gen=gen, offset=offset)
coefsCVmin = coef(path;select=:CVmin)
@test segCVmin == 71
@test coefsCVmin == β[:,segCVmin]

Random.seed!(13)
coefsCV1se = coef(path;select=:CV1se,nCVfolds=20)
segCV1se = cross_validate_path(path,X,y;select=:CV1se,gen=gen,offset=offset)
@test segCV1se == 42
@test coefsCV1se == β[:,segCV1se]

end
