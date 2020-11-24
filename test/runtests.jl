using Test, Lasso, StableRNGs

const testrng = StableRNG(13)

using CSV, DataFrames
readcsvmat(path;header=false, kwargs...) = convert(Matrix{Float64}, DataFrame(CSV.File(path;header=header, kwargs...)))

@testset "Lasso" begin

include("lasso.jl")
include("gammalasso.jl")
include("fusedlasso.jl")
include("trendfiltering.jl")
include("cross_validation.jl")
include("segselect.jl")

end
