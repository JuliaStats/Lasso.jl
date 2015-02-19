using Lasso, FactCheck
DATADIR = joinpath(dirname(@__FILE__), "data")

function diffmatslow(order, n)
    D = spdiagm((fill(-1., n-1), fill(1., n-1)), (0, 1))
    for i = 1:order
        D = spdiagm((fill(-1., n-i-1), fill(1., n-i-1)), (0, 1))*D
    end
    D
end

# Test that DifferenceMatrix behaves as expected
facts("DifferenceMatrix") do
    for order in (1, 2, 3)
        context("order = $(order)") do
            D1 = diffmatslow(order, 100)
            D2 = Lasso.TrendFiltering.DifferenceMatrix{Float64}(order, 100)
            x = randn(100)
            @fact D2'D2 => D1'D1
            @fact D2*x => roughly(D1*x)
            @fact D2'*x[1:size(D1, 1)] => roughly(D1'*x[1:size(D1, 1)])
        end
    end
end

# Test against results from glmgen
lakehuron = readcsv(joinpath(DATADIR, "LakeHuron.csv"); header=true)[1][:, 3]
facts("TrendFiltering") do
    for order in (1, 2, 3)
        context("order = $(order)") do
            for lambda in (1., 10., 100.)
                context("lambda = $(lambda)") do
                    @fact coef(fit(TrendFilter, lakehuron, order, lambda; tol=1e-9)) =>
                          roughly(vec(readcsv(joinpath(DATADIR, "LakeHuron_order_$(order)_lambda_$(lambda).csv"))))
                end
            end
        end
    end
end


