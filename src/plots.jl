using Gadfly, DataFrames, Compat
function Gadfly.plot(path::RegularizationPath, gadfly_args...;
    x=:segment, varnames=nothing, selectedvars=[], select=:AICc, showselectors=[:AICc,:CVmin,:CV1se], nCVfolds=10)
    β=coef(path)
    if hasintercept(path)
        β = β[2:end,:]
    end

    (p,nλ)=size(β)

    if varnames==nothing
        varnames=[symbol("x$i") for i=1:p]
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
    # xintercept = Float64[]
    dashed_vlines=Float64[]
    solid_vlines=Float64[]

    if select == :AICc || :AICc in showselectors
        minAICcix=minAICc(path)
        if select == :AICc
            push!(solid_vlines,indata[minAICcix,x])
        else
            push!(dashed_vlines,indata[minAICcix,x])
        end
    end

    if select == :CVmin || :CVmin in showselectors
        gen = Kfold(length(path.m.rr.y),nCVfolds)
        segCVmin = cross_validate_path(path;gen=gen,select=:CVmin)
        if select == :CVmin
            push!(solid_vlines,indata[segCVmin,x])
        else
            push!(dashed_vlines,indata[segCVmin,x])
        end
    end

    if select == :CV1se || :CV1se in showselectors
        gen = Kfold(length(path.m.rr.y),nCVfolds)
        segCV1se = cross_validate_path(path;gen=gen,select=:CV1se)
        if select == :CV1se
            push!(solid_vlines,indata[segCV1se,x])
        else
            push!(dashed_vlines,indata[segCV1se,x])
        end
    end

    if length(selectedvars) == 0
        if select == :all
            selectedvars = 1:p
        elseif select == :AICc
            selectedvars = find(β[:,minAICcix].!=0)
        elseif select == :CVmin
            selectedvars = find(β[:,segCVmin].!=0)
        elseif select == :CV1se
            selectedvars = find(β[:,segCV1se].!=0)
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
    if length(dashed_vlines) > 0
        append!(layers,layer(xintercept=dashed_vlines, Geom.vline, Theme(default_color=colorant"black",line_style=Gadfly.get_stroke_vector(:dot))))
    end
    if length(solid_vlines) > 0
        append!(layers,layer(xintercept=solid_vlines, Geom.vline, Theme(default_color=colorant"black")))
    end
    if size(inmdframe,1) > 0
      append!(layers, layer(inmdframe,x=x,y="coefficients",color="variable",Geom.line))
    end
    if size(outmdframe,1) > 0
      append!(layers,layer(outmdframe,x=x,y="coefficients",group="variable",Geom.line,Theme(default_color=colorant"lightgray")))
    end

    Gadfly.plot(layers..., Stat.xticks(coverage_weight=1.0), gadfly_args...)
end
