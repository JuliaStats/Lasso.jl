using Gadfly, DataFrames
function Gadfly.plot(path::RegularizationPath, gadfly_args...; x=:segment, varnames=nothing, selectedvars=:nonzeroatAICc, showminAICc=true)
    β=coef(path)
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

    minAICcix=minAICc(path)
    if selectedvars==:nonzeroatAICc || selectedvars==:all
        for j=1:p
            if selectedvars==:all || β[j,minAICcix]!=0
                indata[varnames[j]]=vec(full(β[j,:]))
            else
                outdata[varnames[j]]=vec(full(β[j,:]))
            end
        end
    elseif typeof(selectedvars)==Vector{Int}
        for j in selectedvars
            indata[varnames[j]]=vec(full(β[j,:]))
        end
        for j in setdiff(1:p,selectedvars)
            outdata[varnames[j]]=vec(full(β[j,:]))
        end
    end

    inmdframe=melt(indata,x)
    outmdframe=melt(outdata,x)
    rename!(inmdframe,:value,:β)
    rename!(outmdframe,:value,:β)
    inmdframe = inmdframe[convert(BitArray,map(b->!isnan(b),inmdframe[:β])),:]
    outmdframe = outmdframe[convert(BitArray,map(b->!isnan(b),outmdframe[:β])),:]

    xintercept = []
    if showminAICc
        push!(xintercept,indata[minAICcix,x])
    end

    layers=Vector{Layer}()
    if size(inmdframe,1) > 0
      append!(layers, layer(inmdframe,x=x,y="β",color="variable",Geom.line,xintercept=xintercept,Geom.vline(color=colorant"black")))
    end
    if size(outmdframe,1) > 0
      append!(layers,layer(outmdframe,x=x,y="β",group="variable",Geom.line,Theme(default_color=colorant"lightgray")))
    end

    Gadfly.plot(layers..., Stat.xticks(coverage_weight=1.0), gadfly_args...)
end
