using Gadfly, DataFrames, Compat
function Gadfly.plot(path::RegularizationPath, gadfly_args...;
    x=:segment, varnames=nothing, selectedvars=[], select=:AICc, showselectors=[:AICc,:CVmin,:CV1se], nCVfolds=10)
    β=coef(path)
    if hasintercept(path)
        β = β[2:end,:]
    end

    (p,nλ)=size(β)

    if varnames==nothing
        varnames=[symbol("var$i") for i=1:p]
    end

    indata=DataFrame()
    if x==:λ
        indata[x]=path.λ
    elseif x==:logλ
        indata[x]=log(path.λ)
    else
        x=:segment
        indata[x]=1:nλ
    end
    outdata = deepcopy(indata)

    # CVfun = eval(select)
    # CVfun(oosdevs)

    # automatic selectors
    xintercept = Float64[]

    if select == :AICc || :AICc in showselectors
        minAICcix=minAICc(path)
        push!(xintercept,indata[minAICcix,x])
    end

    if select == :CVmin || :CVmin in showselectors
        gen = Kfold(length(path.m.rr.y),nCVfolds)
        segCVmin = cross_validate_path(path;gen=gen,select=:CVmin)
        push!(xintercept,indata[segCVmin,x])
    end

    if select == :CV1se || :CV1se in showselectors
        gen = Kfold(length(path.m.rr.y),nCVfolds)
        segCV1se = cross_validate_path(path;gen=gen,select=:CV1se)
        push!(xintercept,indata[segCV1se,x])
    end

    if length(selectedvars) == 0
        if select == :all
            selectedvars = 1:p
        elseif select == :AICc
            selectedvars = β[:,minAICcix].!=0
        elseif select == :CVmin
            selectedvars = β[:,segCVmin].!=0
        elseif select == :CV1se
            selectedvars = β[:,segCV1se].!=0
        else
            error("unknown selector $select")
        end
    end

    # colored paths
    for j in selectedvars
        indata[varnames[j]]=vec(full(β[j,:]))
    end

    # grayed out paths
    for j in setdiff(1:p,selectedvars)
        outdata[varnames[j]]=vec(full(β[j,:]))
    end

    inmdframe=melt(indata,x)
    outmdframe=melt(outdata,x)
    rename!(inmdframe,:value,:coefficients)
    rename!(outmdframe,:value,:coefficients)
    inmdframe = inmdframe[convert(BitArray,map(b->!isnan(b),inmdframe[:coefficients])),:]
    outmdframe = outmdframe[convert(BitArray,map(b->!isnan(b),outmdframe[:coefficients])),:]

    layers=@compat Vector{Layer}()
    if size(inmdframe,1) > 0
      append!(layers, layer(inmdframe,x=x,y="coefficients",color="variable",Geom.line,xintercept=xintercept,Geom.vline(color=colorant"black")))
    end
    if size(outmdframe,1) > 0
      append!(layers,layer(outmdframe,x=x,y="coefficients",group="variable",Geom.line,Theme(default_color=colorant"lightgray")))
    end

    Gadfly.plot(layers..., Stat.xticks(coverage_weight=1.0), gadfly_args...)
end
