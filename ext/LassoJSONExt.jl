module LassoJSONExt

using Lasso
using Lasso: RegularizedModel, InferenceModel, to_dict
using JSON

function Lasso.write_json(io::IO, m::RegularizedModel)
    return JSON.print(io, to_dict(m))
end

function Lasso.write_json(path::AbstractString, m::RegularizedModel)
    open(path, "w") do io
        Lasso.write_json(io, m)
    end
end

function Lasso.read_json(io::IO)
    return InferenceModel(JSON.parse(io))
end

function Lasso.read_json(path::AbstractString)
    open(path, "r") do io
        Lasso.read_json(io)
    end
end

end
