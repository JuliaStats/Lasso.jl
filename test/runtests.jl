using Test, Lasso

using CSV
readcsvmat(path;header=false, kwargs...) = convert(Matrix{Float64},CSV.read(path;header=header, kwargs...))

@testset "Lasso" begin

include("lasso.jl")
include("gammalasso.jl")
include("fusedlasso.jl")
include("trendfiltering.jl")
include("cross_validation.jl")
include("segselect.jl")

end
