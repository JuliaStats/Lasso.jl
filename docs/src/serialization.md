# Saving and loading a fitted model

A fitted `RegularizedModel` (e.g. a `LassoModel`) can be reduced to the
coefficients, intercept flag, link, and offset requirement needed to call
`predict`, without keeping the full fit and its underlying data around. This
is done via [`InferenceModel`](@ref).

`InferenceModel(m)` extracts this state from a fitted model (also works on
the `TableRegressionModel` wrapper returned by formula-based `fit`, as used
below). `to_dict` converts an `InferenceModel` to a plain `Dict`, and
`InferenceModel(d)` reconstructs one from such a `Dict`:

```jldoctest
julia> using DataFrames, Lasso

julia> data = DataFrame(X=[1,2,3], Y=[2,4,7]);

julia> m = fit(LassoModel, @formula(Y ~ X), data; select=MinAICc());

julia> m2 = Lasso.InferenceModel(m);

julia> d = Lasso.to_dict(m2);

julia> m3 = Lasso.InferenceModel(d);

julia> predict(m3, reshape(data.X, :, 1)) ≈ predict(m, data)
true
```

If the original model was fit with an `offset`, `predict(m2, newX; offset=...)`
requires it, matching `predict(m::RegularizedModel, newX; offset=...)`.
`InferenceModel` only supports a single fitted `LassoModel`/
`GammaLassoModel` segment — calling it on a full `LassoPath`/`GammaLassoPath`
or on a `FusedLasso`/`TrendFilter` model raises an `ArgumentError`.

This roundtrip doesn't depend on any particular serialization format. Loading
`JSON.jl` alongside Lasso.jl also enables `write_json`/`read_json`, which
build on `InferenceModel`/`to_dict` to save to and load from JSON directly
(and, like `InferenceModel`, accept a fitted model directly, not just an
`InferenceModel`):

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
