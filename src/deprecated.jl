import Base: @deprecate

# """
#     coef(path::RegularizationPath; select=:AICc)
#
# Returns a p by nλ coefficient array where p is the number of
# coefficients (including any intercept) and nλ is the number of path segments,
# or a selected segment's coefficients.
#
# If model was only initialized but not fit, returns a p vector of zeros.
# Consistent with StatsBase.coef, if the model has an intercept it is included.
# 
# # Example:
# ```julia
#   m = fit(LassoPath,X,y)
#   coef(m; select=:CVmin)
# ```
#
# # Keywords
# - `select=:all` returns a p by nλ matrix of coefficients
# - `select=:AIC` selects the AIC minimizing segment
# - `select=:AICc` selects the corrected AIC minimizing segment
# - `select=:BIC` selects the BIC minimizing segment
# - `select=:CVmin` selects the segment that minimizing out-of-sample mean squared
#     error from cross-validation with `nCVfolds` random folds.
# - `select=:CV1se` selects the segment whose average OOS deviance is no more than
#     1 standard error away from the one minimizing out-of-sample mean squared
#     error from cross-validation with `nCVfolds` random folds.
# - `nCVfolds=10` number of cross-validation folds
# - `kwargs...` are passed to [`minAICc(::RegularizationPath)`](@ref) or to
#     [`cross_validate_path(::RegularizationPath)`](@ref)
# """
function StatsBase.coef(path::RegularizationPath, select::Symbol; nCVfolds=10)
    if select == :all
        selector = AllSeg()
        msg = "AllSeg()"
    elseif select == :AIC
        selector = MinAIC()
        msg = "MinAIC()"
    elseif select == :AICc
        selector = MinAICc()
        msg = "MinAICc()"
    elseif select == :BIC
        selector = MinBIC()
        msg = "MinBIC()"
    elseif select == :CVmin
        selector = MinCVmse(path, nCVfolds)
        msg = "MinCVmse(path, nCVfolds)"
    elseif select == :CV1se
        selector = MinCV1se(path, nCVfolds)
        msg = "MinCV1se(path, nCVfolds)"
    else
        error("unknown selector $select")
    end

    Base.depwarn("coef(path; select::Symbol=:$select) is deprecated, use coef(path, $msg) instead", :coef)

    coef(path, selector)
end
