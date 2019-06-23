using Documenter, Lasso, MLBase

makedocs(
    modules = [Lasso],
    format = Documenter.HTML(
        # Use clean URLs, unless built as a "local" build
        prettyurls = !("local" in ARGS),
        canonical = "https://juliastats.github.io/Lasso.jl/stable/",
    ),
    clean = false,
    sitename = "Lasso.jl",
    authors = "Simon Kornblith, Asaf Manela, and contributors.",
    linkcheck = !("skiplinks" in ARGS),
    pages = [
        "Home" => "index.md",
        "Lasso paths" => "lasso.md",
        "Fused Lasso and trend filtering" => "smoothing.md",
        "Index" => "api.md",
    ],
    )

deploydocs(
    repo = "github.com/JuliaStats/Lasso.jl.git",
    target = "build",
)
