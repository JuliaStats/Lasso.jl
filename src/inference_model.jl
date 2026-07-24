"""
InferenceModel stores the minimal state of a fitted `RegularizedModel` needed to
call `predict`: coefficients, whether an intercept was fit, the link function,
and whether the model requires an `offset` at predict time.

Constructed from a fitted model via `InferenceModel(m)` (works on a
`LassoModel`/`GammaLassoModel`, or its formula-fit `TableRegressionModel`
wrapper) or from a `Dict` via `InferenceModel(d)` (see [`to_dict`](@ref), or
[`read_json`](@ref)).
"""
struct InferenceModel
    coef::Vector{Float64}
    intercept::Bool
    link::Link
    hasoffset::Bool
end

InferenceModel(m::InferenceModel) = m

"""
    InferenceModel(m::RegularizedModel) -> InferenceModel
    InferenceModel(m::StatsModels.TableRegressionModel{<:RegularizedModel}) -> InferenceModel

Extract the coefficients, intercept flag, link, and offset requirement of a
fitted `RegularizedModel` (e.g. a `LassoModel`), sufficient to call `predict`
without keeping the full fit and its underlying data around.
"""
function InferenceModel(m::RegularizedModel)
    link = _get_link(m.lpm)
    haskey(LINK_TYPE_TO_NAME, typeof(link)) ||
        error("Unsupported link type for serialization: $(typeof(link))")
    return InferenceModel(coef(m), GLM.hasintercept(m.lpm), link, !isempty(m.lpm.rr.offset))
end

InferenceModel(mm::StatsModels.TableRegressionModel{<:RegularizedModel}) = InferenceModel(mm.model)

InferenceModel(path::RegularizationPath) = throw(ArgumentError(
    "InferenceModel is not defined for a full $(nameof(typeof(path))); it only supports a " *
    "single selected segment. Use `fit(LassoModel, ...)` or `fit(GammaLassoModel, ...)` to " *
    "fit and select one segment, then call `InferenceModel` on that."))

InferenceModel(m) = throw(ArgumentError(
    "InferenceModel is not defined for $(typeof(m)); only a fitted LassoModel/GammaLassoModel " *
    "segment (or its formula-fit TableRegressionModel wrapper) is supported."))

"link of the underlying GLM of a fitted RegularizedModel segment"
_get_link(lpm::LinearModel) = IdentityLink()
_get_link(lpm::GeneralizedLinearModel) = Link(lpm)

# explicit type <-> name tables
# This deliberately does not use metaprogramming or `nameof` or similar,
# because if the julia type `IdentityLink` is renamed to e.g. `IdentityLink2`,
# we want to continue to read/write it as the string
# "IdentityLink" for stability in the serialized format.
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
    to_dict(m::InferenceModel) -> Dict

Convert an [`InferenceModel`](@ref) to a plain `Dict`, sufficient to
reconstruct it via `InferenceModel(d)`.
"""
function to_dict(m::InferenceModel)
    linkname = get(LINK_TYPE_TO_NAME, typeof(m.link)) do
        error("Unsupported link type for serialization: $(typeof(m.link))")
    end
    return Dict(
        "coef" => m.coef,
        "intercept" => m.intercept,
        "link" => linkname,
        "hasoffset" => m.hasoffset,
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
    return InferenceModel(Float64.(d["coef"]), d["intercept"], constructor(), d["hasoffset"])
end

"""
    predict(m::InferenceModel, newX::AbstractMatrix; offset=eltype(newX)[])

Predicted values from an `InferenceModel`. If the original model was fit with
an offset, `offset` must be supplied here with one value per row of `newX`
(mirroring `predict(m::RegularizedModel, newX; offset=...)`).
"""
function StatsBase.predict(m::InferenceModel, newX::AbstractMatrix{T};
                            offset::AbstractVector{<:Real}=T[]) where T
    X = m.intercept ? [ones(T, size(newX, 1), 1) newX] : newX
    eta = X * m.coef
    if m.hasoffset
        length(offset) == size(newX, 1) ||
            throw(ArgumentError("model was fit with an offset, so `offset` kwarg must have length size(newX, 1)"))
        eta = eta .+ offset
    else
        isempty(offset) ||
            throw(ArgumentError("model was fit without an offset, so the `offset` kwarg does not make sense"))
    end
    linkinv.(m.link, eta)
end

"""
    write_json(io::IO, m)
    write_json(path::AbstractString, m)

Save the coefficients, intercept flag, link, and offset requirement of a
fitted `RegularizedModel` (e.g. a `LassoModel`, or an already-constructed
[`InferenceModel`](@ref)) as JSON, sufficient to later call `predict` via
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
