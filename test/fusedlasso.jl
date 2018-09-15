using Lasso
DATADIR = joinpath(dirname(@__FILE__), "data")

# Test against fused lasso results from glmgen
lakehuron = readcsvmat(joinpath(DATADIR, "LakeHuron.csv");header=true)[:, 3]
@testset "FusedLasso" begin
	@testset "λ = $(lambda)" for lambda in (10, 1, 0.1)
		@test coef(fit(FusedLasso, lakehuron, lambda)) ≈ vec(readcsvmat(joinpath(DATADIR, "LakeHuron_lambda_$lambda.csv")))
	end
end
