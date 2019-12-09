using Lasso, GLM, DataFrames, StatsModels

X = rand(9,11)
y = rand(9)

# this is to run this step: https://github.com/JuliaStats/Lasso.jl/blob/9d099f512afb6042b1094714e0ab280cf10f3863/src/segselect.jl#L276
#   I set λ to a single variable to simplify things
path = fit(LassoPath, X,y; λ=[0.005])

m1 = selectmodel(path, MinAICc());

coef(m1)
stderror(m1)
typeof(m1)
predict(m1)

m2 = selectmodel(path, MinAICc());
coef(m2)

m3 = selectmodel(path, MinAICc());
coef(m3)

lm1 = fit(LassoModel, X,y; select=MinAICc(), λ=[0.005])
vcov(lm1)
stderror(lm1)

df = DataFrame([X y])
rename!(df, names(df)[end]=>:y)
f = @formula(y ~ x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11)
lm2 = fit(LassoModel, f, df; select=MinAICc(), λ=[0.005])
coef(lm2) == coef(lm1)
predict(lm2)
predict(lm2, df)