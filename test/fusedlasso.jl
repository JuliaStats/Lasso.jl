using Lasso, FactCheck
const DATADIR = joinpath(dirname(@__FILE__), "data")

lakehuron = readcsv(joinpath(DATADIR, "LakeHuron.csv"); header=true)[1][:, 3]

# Test against fused lasso results from glmgen
facts("FusedLasso") do
	for lambda in (10, 1, 0.1)
		context("λ = $(lambda)") do
			@fact coef(fit(FusedLasso, lakehuron, lambda)) => roughly(vec(readcsv(joinpath(DATADIR, "LakeHuron_lambda_$lambda.csv"))))
		end
	end
end

