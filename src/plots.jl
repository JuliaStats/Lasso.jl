using Gadfly, DataFrames
import Gadfly.plot
function plot(path::RegularizationPath; x=:iteration, varnames=nothing, selectedvars=:nonzero)
    β=coef(path)
    (p,nλ)=size(β)

    if varnames==nothing
        varnames=[symbol("var$i") for i=1:p]
    end

    df=DataFrame()
    if x==:lambda
        df[x]=path.λ
    elseif x==:loglambda
        df[x]=log(path.λ)
    else
        df[x]=1:nλ
    end

    if selectedvars==:nonzero || selectedvars==:all
        for j=1:p
            if selectedvars==:all || !all(β[j,:].==0)
                df[varnames[j]]=vec(full(β[j,:]))
            end
        end
    elseif typeof(selectedvars)==Vector{Int}
        for j in selectedvars
            df[varnames[j]]=vec(full(β[j,:]))
        end
    end

    df=melt(df,x)
    rename!(df,:value,:β)
    Gadfly.plot(df,x=x,y="β",color="variable",Geom.line,
     Stat.xticks(coverage_weight=1.0))
end
