## SERIALIZATION-INDEPENDENT SUPPORT FOR SAVING/LOADING A FITTED MODEL

"""
InferenceModel stores the minimal state of a fitted `RegularizedModel` needed to
call `predict`: coefficients, whether an intercept was fit, and the link
function.

Constructed via `InferenceModel(d::Dict)` (or [`read_json`](@ref)); the
corresponding fitted model is converted via [`to_dict`](@ref) (or saved via
[`write_json`](@ref)).
"""
struct InferenceModel
    coef::Vector{Float64}
    intercept::Bool
    link::Link
end

"link of the underlying GLM of a fitted RegularizedModel segment"
_get_link(lpm::LinearModel) = IdentityLink()
_get_link(lpm::GeneralizedLinearModel) = lpm.rr.link

# explicit type <-> name tables, rather than nameof/eval, so the serialized
# format is stable across GLM renames and InferenceModel(d) can only ever
# construct one of these known link types
const LINK_TYPE_TO_NAME = Dict(
    IdentityLink          => "IdentityLink",
    LogitLink             => "LogitLink",
    LogLink               => "LogLink",
    InverseLink           => "InverseLink",
    CloglogLink           => "CloglogLink",
    CauchitLink           => "CauchitLink",
    ProbitLink            => "ProbitLink",
    SqrtLink              => "SqrtLink",
    InverseSquareLink     => "InverseSquareLink",
    NegativeBinomialLink  => "NegativeBinomialLink",
)
const LINK_NAME_TO_CONSTRUCTOR = Dict(name => T for (T, name) in LINK_TYPE_TO_NAME)

"""
    to_dict(m::RegularizedModel) -> Dict

Convert the coefficients, intercept flag, and link of a fitted
`RegularizedModel` (e.g. a `LassoModel`) to a plain `Dict`, sufficient to
reconstruct an [`InferenceModel`](@ref) via `InferenceModel(d)`.
"""
function to_dict(m::RegularizedModel)
    link = _get_link(m.lpm)
    linkname = get(LINK_TYPE_TO_NAME, typeof(link)) do
        error("Unsupported link type for serialization: $(typeof(link))")
    end
    return Dict(
        "coef" => coef(m),
        "intercept" => m.intercept,
        "link" => linkname,
    )
end

"""
    InferenceModel(d::AbstractDict) -> InferenceModel

Reconstruct an [`InferenceModel`](@ref) from a `Dict` produced by
[`to_dict`](@ref).
"""
function InferenceModel(d::AbstractDict)
    linkname = d["link"]
    constructor = get(LINK_NAME_TO_CONSTRUCTOR, linkname) do
        error("Unsupported link name: $linkname")
    end
    return InferenceModel(Float64.(d["coef"]), d["intercept"], constructor())
end

function StatsBase.predict(m::InferenceModel, newX::AbstractMatrix{T}) where T
    X = m.intercept ? [ones(T, size(newX, 1), 1) newX] : newX
    linkinv.(m.link, X * m.coef)
end

"""
    write_json(io::IO, m::RegularizedModel)
    write_json(path::AbstractString, m::RegularizedModel)

Save the coefficients, intercept flag, and link of a fitted `RegularizedModel`
(e.g. a `LassoModel`) as JSON, sufficient to later call `predict` via
[`read_json`](@ref). Requires JSON.jl to be loaded.
"""
function write_json end

"""
    read_json(io::IO) -> InferenceModel
    read_json(path::AbstractString) -> InferenceModel

Load a model saved by [`write_json`](@ref) as an [`InferenceModel`](@ref),
which supports `predict`. Requires JSON.jl to be loaded.
"""
function read_json end
