## JSON-INDEPENDENT SUPPORT FOR SAVING/LOADING A FITTED MODEL

"""
JSONModel stores the minimal state of a fitted `RegularizedModel` needed to call
`predict`: coefficients, whether an intercept was fit, and the link function.

Constructed via [`read_json`](@ref); the corresponding fitted model is saved via
[`write_json`](@ref). Both require JSON.jl to be loaded.
"""
struct JSONModel
    coef::Vector{Float64}
    intercept::Bool
    link::Link
end

"link of the underlying GLM of a fitted RegularizedModel segment"
_get_link(lpm::LinearModel) = IdentityLink()
_get_link(lpm::GeneralizedLinearModel) = lpm.rr.link

function StatsBase.predict(m::JSONModel, newX::AbstractMatrix{T}) where T
    X = m.intercept ? [ones(T, size(newX, 1), 1) newX] : newX
    linkinv.(m.link, X * m.coef)
end

"""
    write_json(io::IO, m::RegularizedModel)
    write_json(path::AbstractString, m::RegularizedModel)

Save the coefficients, intercept flag, and link of a fitted `RegularizedModel`
(e.g. a `LassoModel`) as JSON, sufficient to later call `predict` via
[`read_json`](@ref). Requires a JSON package to be loaded.
"""
function write_json end

"""
    read_json(io::IO) -> JSONModel
    read_json(path::AbstractString) -> JSONModel

Load a model saved by [`write_json`](@ref) as a [`JSONModel`](@ref), which
supports `predict`. Requires a JSON package to be loaded.
"""
function read_json end
