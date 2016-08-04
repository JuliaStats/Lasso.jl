using MLBase
function cross_validate_path{T}(estfun::Function, evalfun::Function, nobs::Int, λ::Vector{T}, gen)
    nfolds = length(gen)
    nλ = length(λ)
    scores = zeros(T,nλ,nfolds)
    for (j, train_inds) in enumerate(gen)
        test_inds = setdiff(1:nobs, train_inds)
        for i=1:nλ
          model = estfun(train_inds)
          score = evalfun(model, test_inds)
          scores[i,j] = score
      end
    end
    scores
end
