"RegularizationPath segment selector supertype"
abstract type SegSelect end

"Index of the selected RegularizationPath segment"
segselect(path::RegularizationPath, select::S) where S<:SegSelect =
    throw("segselect(path, ::$S) is not implemented")

"A RegularizationPath segment selector that returns all segments"
struct AllSeg <: SegSelect end

"Selects the RegularizationPath segment with the minimum AIC"
struct MinAIC <: SegSelect end

"Index of the selected RegularizationPath segment"
segselect(path::RegularizationPath, select::MinAIC) = minAIC(path)

"Selects the RegularizationPath segment with the minimum corrected AIC"
struct MinAICc <: SegSelect
    k::Int # k parameter used to correct AIC criterion
    MinAICc(k::Int=2) = new(k)
end
segselect(path::RegularizationPath, select::MinAICc) = minAICc(path; k=select.k)

"Selects the RegularizationPath segment with the minimum BIC"
struct MinBIC <: SegSelect end

segselect(path::RegularizationPath, select::MinBIC) = minBIC(path)

"RegularizationPath segment selector supertype"
abstract type CVSegSelect <: SegSelect end

"Selects the RegularizationPath segment with the minimum cross-validation mse"
struct MinCVmse <: CVSegSelect
    gen::CrossValGenerator

    MinCVmse(gen::CrossValGenerator) = new(gen)
    MinCVmse(path::RegularizationPath, k::Int=10) = new(Kfold(length(path.m.rr.y), k))
end

CVfun(oosdevs, ::MinCVmse) = CVmin(oosdevs)

"""
Selects the RegularizationPath segment with the largest λt with mean
OOS deviance no more than one standard error away from minimum
"""
struct MinCV1se <: CVSegSelect
    gen::CrossValGenerator

    MinCV1se(gen::CrossValGenerator) = new(gen)
    MinCV1se(path::RegularizationPath, k::Int=10) = new(Kfold(length(path.m.rr.y), k))
end

CVfun(oosdevs, ::MinCV1se) = CV1se(oosdevs)

StatsBase.coef(path::RegularizationPath; select=AllSeg(), kwargs...) = coef(path, select; kwargs...)
function StatsBase.coef(path::RegularizationPath, select::S) where S <: SegSelect
    if !isdefined(path,:coefs)
        X = path.m.pp.X
        p,nλ = size(path)
        return zeros(eltype(X),p)
    end

    seg = segselect(path, select)

    if hasintercept(path)
        vec(vcat(path.b0[seg],path.coefs[:,seg]))
    else
        path.coefs[:,seg]
    end
end

function StatsBase.coef(path::RegularizationPath, select::AllSeg)
    if !isdefined(path,:coefs)
        X = path.m.pp.X
        p,nλ = size(path)
        return spzeros(eltype(X),p,nλ)
    end

    if hasintercept(path)
        vcat(path.b0',path.coefs)
    else
        path.coefs
    end
end

segselect(path::RegularizationPath, select::S) where S<:CVSegSelect =
    cross_validate_path(path, select)

segselect(path::RegularizationPath,
           X::AbstractMatrix{T}, y::V,        # potentially new data
           select::S;
           kwargs...) where {T<:AbstractFloat,V<:FPVector, S<:CVSegSelect} =
    cross_validate_path(path, X, y, select; kwargs...)

"A RegularizedModel represents selected segment from a RegularizationPath"
abstract type RegularizedModel <: RegressionModel end

"A LassoModel represents selected segment from a LassoPath"
abstract type LassoModel <: RegularizedModel end

"A GammaLassoModel represents selected segment from a GammaLassoPath"
abstract type GammaLassoModel <: RegularizedModel end

"Returns the RegularizedPath type R used in fit(R,...)"
pathtype(::Type{LassoModel}) = LassoPath
pathtype(::Type{GammaLassoModel}) = GammaLassoPath

function selectmodel(path::R, select::SegSelect) where R<:RegularizationPath
    # extract reusable path parts
    m = path.m
    pp = m.pp
    X = pp.X

    # add an interecept to X if the model has one
    if hasintercept(path)
        segX = [ones(eltype(X),size(X,1),1) X]
    end

    # select coefs
    beta0 = coef(path, select)

    # create new linear predictor
    segpp = DensePredQR(segX, beta0)

    # create a LinearModel or GeneralizedLinearModel with the new linear predictor
    newglm(m, segpp)
end

function StatsBase.fit(::Type{R}, args...;
    select::SegSelect=MinAICc(), kwargs...) where R<:RegularizedModel

    # fit a regularization path
    M = pathtype(R)
    path = fit(M, args...; kwargs...)

    selectmodel(path, select)
end

newglm(m::LinearModel, pp) = LinearModel(m.rr, pp)
newglm(m::GeneralizedLinearModel, pp) = GeneralizedLinearModel(m.rr, pp, true)
