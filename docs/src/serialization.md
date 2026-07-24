# Saving and loading a fitted model

A fitted `RegularizedModel` (e.g. a `LassoModel`) can be reduced to the
coefficients, intercept flag, link, and offset requirement needed to call
`predict`, without keeping the full fit and its underlying data around. This
is done via [`InferenceModel`](@ref).

`to_dict` converts a fitted model to a plain `Dict`, and `InferenceModel`
can be reconstructed from one:

```jldoctest
julia> using DataFrames, Lasso

julia> data = DataFrame(X=[1,2,3], Y=[2,4,7]);

julia> m = fit(LassoModel, @formula(Y ~ X), data; select=MinAICc());

julia> d = Lasso.to_dict(m);

julia> m2 = Lasso.InferenceModel(d);

julia> predict(m2, reshape(data.X, :, 1)) ≈ predict(m, data)
true
```

`to_dict`/`write_json` also work directly on the `TableRegressionModel` wrapper
returned by formula-based `fit`, as used above.

If the original model was fit with an `offset`, `predict(m2, newX; offset=...)`
requires it, matching `predict(m::RegularizedModel, newX; offset=...)`.
`to_dict`/`write_json` only support a single fitted `LassoModel`/
`GammaLassoModel` segment — calling them on a full `LassoPath`/`GammaLassoPath`
or on a `FusedLasso`/`TrendFilter` model raises an `ArgumentError`.

This roundtrip doesn't depend on any particular serialization format. Loading
`JSON.jl` alongside Lasso.jl also enables
`write_json`/`read_json`, which build on `to_dict`/`InferenceModel` to save
to and load from JSON directly:

```julia
using Lasso, JSON

write_json("model.json", m)
m2 = read_json("model.json")
predict(m2, newX) ≈ predict(m, newX)
```

```@docs
InferenceModel
to_dict
write_json
read_json
```
