module LassoJSONExt

using Lasso
using Lasso: RegularizedModel, JSONModel, _get_link
using JSON
using GLM

# explicit type <-> name tables, rather than nameof/eval, so the JSON wire
# format is stable across GLM renames and read_json can only ever construct
# one of these known link types
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
    write_json(io::IO, m::RegularizedModel)
    write_json(path::AbstractString, m::RegularizedModel)

Save the coefficients, intercept flag, and link of a fitted `RegularizedModel`
(e.g. a `LassoModel`) to a JSON file at `path`.
"""
function Lasso.write_json(io::IO, m::RegularizedModel)
    link = _get_link(m.lpm)
    linkname = get(LINK_TYPE_TO_NAME, typeof(link)) do
        error("Unsupported link type for JSON serialization: $(typeof(link))")
    end
    d = Dict(
        "coef" => coef(m),
        "intercept" => m.intercept,
        "link" => linkname,
    )
    return JSON.print(io, d)
end

function Lasso.write_json(path::AbstractString, m::RegularizedModel)
    open(path, "w") do io
        Lasso.write_json(io, m)
    end
end

"""
    read_json(io::IO) -> JSONModel
    read_json(path::AbstractString) -> JSONModel

Load a model saved by [`write_json`](@ref) as a `JSONModel`, which supports
`predict`.
"""
function Lasso.read_json(io::IO)
    d = JSON.parse(io)
    linkname = d["link"]
    constructor = get(LINK_NAME_TO_CONSTRUCTOR, linkname) do
        error("Unsupported link name in JSON file: $linkname")
    end
    return JSONModel(Float64.(d["coef"]), d["intercept"], constructor())
end

function Lasso.read_json(path::AbstractString)
    open(path, "r") do io
        Lasso.read_json(io)
    end
end

end
