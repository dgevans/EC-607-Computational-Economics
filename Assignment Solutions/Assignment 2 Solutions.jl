using Roots,QuantEcon,NLsolve,Parameters,Gadfly,DataFrames,LinearAlgebra


@with_kw mutable struct RBCmodelNS
    β::Float64 
    α::Float64 
    θ::Float64 
    δ::Float64 

    #steady state
    h̄::Float64 
    ī::Float64
    k̄::Float64
    c̄::Float64

    #objects for Bellman equation
    kgrid::Vector{Float64} = zeros(0)
    U::Matrix{Float64} = zeros(0,0)
    h::Matrix{Float64} = zeros(0,0)
end

"""
    RBCmodelNS()

Constructs RBCmodelNS based on calibration and stores steady state
"""
function RBCmodelNS()
    β = 0.98
    α = 0.3
    #first compute steady state
    h̄ = 0.3
    δ = (1/β-1)/(α/0.16 - 1)
    k̄ = (δ/0.16*0.3^(α-1))^(1/(α-1))
    c̄ = k̄^α*h̄^(1-α)-δ*k̄
    θ = (1 + (1-h̄)*(1-α)*k̄^α*h̄^(-α)/c̄)^(-1)
    ī = δ*k̄
    return RBCmodelNS(β=β,α=α,θ=θ,δ=δ,h̄=h̄,ī=ī,k̄=k̄,c̄=c̄)
end

"""
    path_residuals(model::RBCmodelNS,k0::Float64,ihpath)

compute residuals given a path of i and h
"""
function path_residuals!(model::RBCmodelNS,k0::Float64,res,ihpath,T)
    @unpack α,β,δ,θ,k̄,h̄,c̄ = model
    ipath,hpath = ihpath[1:T],ihpath[T+1:end]
    k = zeros(T+2)
    k[1] = k0
    for t in 1:T
        k[t+1] = (1-δ)*k[t] + ipath[t] 
    end
    k[T+2] = k̄
    h = [hpath;h̄]
    #compute path of c given guess of $k$
    c = [k[t]^α*h[t]^(1-α) + (1-δ)*k[t] - k[t+1] for t in 1:T+1]
    u_c = @. θ/c
    u_n = @. (1-θ)/(1 - h)
    F_n = @. (1-α)*k[1:T+1]^α*h^(-α)
    R = @. 1-δ+α*k[1:T+1]^(α-1)*h^(1-α)

    res[1:T] .= u_c[1:T].-β.*R[2:T+1].*u_c[2:T+1]
    res[T+1:end] .= u_c[1:T].*F_n[1:T].-u_n[1:T]
end 

"""
    solve_transition(model::RBCmodelNS,k0,T)
    
returns a path of 
"""
function solve_transition(model::RBCmodelNS,k0,T)
    @unpack α,β,δ,θ,k̄,h̄,c̄ = model
    f!(F,x) = path_residuals!(model,k0,F,x,T)
    ihpath0 = [model.ī*ones(T);model.h̄*ones(T)]

    res = nlsolve(f!,ihpath0)

    if !converged(res)
        error("Could not find root")
    end
    #Now back out aggregates
    ihpath = res.zero
    ipath,hpath = ihpath[1:T],ihpath[T+1:end]
    k = zeros(T+2)
    k[1] = k0
    for t in 1:T
        k[t+1] = (1-δ)*k[t] + ipath[t] 
    end
    k[T+2] = k̄
    h = [hpath;h̄]
    #compute path of c given guess of $k$
    c = [k[t]^α*h[t]^(1-α) + (1-δ)*k[t] - k[t+1] for t in 1:T+1]
    y = [k[t]^α*h[t]^(1-α) for t in 1:T+1]
    return DataFrame(k=k[1:T],c=c[1:T],h=h[1:T],i=ipath,y=y[1:T])
end

model = RBCmodelNS()
ihpath = [model.ī*ones(100) model.h̄*ones(100)]


df = solve_transition(model,model.k̄+3,100)

gridstack([plot(df,y=:k,Geom.line) plot(df,y=:c,Geom.line);plot(df,y=:h,Geom.line) plot(df,y=:i,Geom.line)])

"""
    construct_indirectU!(model::RBCmodelNS)

Solves for the indirect utility and hours if entering with assets
kgrid[n] and saving kgrid[n′]
"""
function construct_indirectU!(model::RBCmodelNS)
    @unpack α,β,δ,θ,kgrid = model
    N = length(kgrid)
    U = model.U = zeros(N,N)
    h = model.h = zeros(N,N)

    for n in 1:N
        for n′ in 1:N
            i = kgrid[n′] - (1-δ)*kgrid[n]
            k = kgrid[n]
            if i > k^α - 0.00001 #output cannot cover investment and a small amount of consumption
                U[n,n′] = -Inf
            else
                #minimal labor supply to ensure positive consumption
                hmin = i>0 ? ((i+0.00001)/k^α)^(1/(1-α)) : 0.00001
                f(h) = ((1-α)*k^α*h^(-α)*θ/(k^α*h^(1-α)-i) - (1-θ)/(1-h)) #optimal labor leisure choice
                if f(hmin) <= 0 
                    #investment is negative enough that negative hours solves equation impose h=0
                    h[n,n′] = hmin
                else
                    h[n,n′] = brent(f,hmin,0.999999999999)
                end
                c = k^α*h[n,n′]^(1-α) -i
                U[n,n′] = θ*log(c) + (1-θ)*log(1-h[n,n′])
            end
        end 
    end
end

model.kgrid = LinRange(model.k̄*0.3,model.k̄*1.7,400)

construct_indirectU!(model)


"""
    RBCbellmanmap(model::RBCmodelNS,V′)

Iterates on the bellman equation for the standard neoclassical growth model

# Arguments
* `V` Vector of values for each capital level in kgrid
"""
function RBCbellmanmap(model::RBCmodelNS,V′)
    @unpack kgrid,U,h,β = model
    N = length(kgrid)
    V = zeros(N) #new value function
    n_pol = zeros(Int,N) #policy rule for grid points
    k_pol = zeros(N) #policy rule for capital
    h_pol = zeros(N)
    obj = zeros(N)
    for n in 1:N #iterate for each initial capital
        obj .= U[n,:] .+ β.*V′ #note using indirect utiltiy
        V[n],n_pol[n] = findmax(obj) #find optimal value and the choice that gives it
        k_pol[n] = kgrid[n_pol[n]] #record capital policy
        h_pol[n] = h[n,n_pol[n]] #record hours policy
    end
    return V,n_pol,k_pol,h_pol
end

"""
RBCbellmanmap_howard(model::RBCmodelNS,V′,n_pol)

Iterates on the bellman equation for the standard neoclassical growth model

# Arguments
* `V` Vector of values for each capital level in kgrid
* n_pol capital policy
"""
function RBCbellmanmap_howard(model::RBCmodelNS,V′,n_pol)
    @unpack kgrid,U,h,β = model
    N = length(kgrid)
    V = zeros(N) #new value function
    for n in 1:N #iterate for each initial capital
        V[n] = U[n,n_pol[n]] + β*V′[n_pol[n]] #note using indirect utiltiy
    end
    return V
end

"""
    RBCsolve_bellman(model,V0,[,ϵ=1e-8])

Solves the bellman equation by iterating until convergence

# Arguments
* `V0` Initial vector of values for each capital level in kgrid
* `ϵ` Convergence criteria
"""
function RBCsolve_bellman(model,V0,H=100,ϵ=1e-8)
    diff = 1.
    V,n_pol,k_pol,h_pol = RBCbellmanmap(model,V0)
    while diff > ϵ
        for h in 1:H
            V= RBCbellmanmap_howard(model,V,n_pol)
        end
        V_new,n_pol,k_pol,h_pol = RBCbellmanmap(model,V)
        diff = norm(V_new-V,Inf)
        println(diff)
        V = V_new 
    end
    return V,n_pol,k_pol,h_pol
end

"""
    simulate(model::RBCmodelNS,n_0,T,n_pol,h_pol)

Simulate the path of aggregates for T periods given initial state n_0
"""
function simulate(model::RBCmodelNS,n_0,T,n_pol,h_pol)
    @unpack α,β,δ,θ,kgrid = model
    k = zeros(T+1)
    n = zeros(Int,T+1)
    c,h,y,i = zeros(T),zeros(T),zeros(T),zeros(T)
    n[1] = n_0
    k[1] = kgrid[n_0]
    #no simulate using policies
    for t in 1:T
        n[t+1] = n_pol[n[t]]
        k[t+1] = kgrid[n[t+1]]
        h[t] = h_pol[n[t]]
        y[t] = k[t]^α*h[t]^(1-α)
        i[t] = k[t+1] - (1-δ)*k[t]
        c[t] = y[t] - i[t]
    end

    return DataFrame(k=k[1:T],h=h,c=c,i=i,y=y)
end
V,n_pol,k_pol,h_pol = RBCsolve_bellman(model,zeros(length(model.kgrid)))


n0=400
df = simulate(model,n0,100,n_pol,h_pol)
df2 = solve_transition(model,model.kgrid[n0],100)
plot(layer(df,y=:i,color=["bellman"]),layer(df2,y=:i,color=["Perfect Foresight"]))


"""
    solve_transition_nri(model::RBCmodelNS,k0,T)
    
returns a path of aggregates if investment is non-reversable
"""
function solve_transition_nri(model::RBCmodelNS,k0,T)
    @unpack α,β,δ,θ,k̄,h̄,c̄ = model
    f!(F,x) = path_residuals!(model,k0,F,x,T)
    ihpath0 = [model.ī*ones(T);model.h̄*ones(T)]

    res = mcpsolve(f!,zeros(2*T),Inf*ones(2(T)),ihpath0)

    if !converged(res)
        error("Could not find root")
    end
    #Now back out aggregates
    ihpath = res.zero
    ipath,hpath = ihpath[1:T],ihpath[T+1:end]
    k = zeros(T+2)
    k[1] = k0
    for t in 1:T
        k[t+1] = (1-δ)*k[t] + ipath[t] 
    end
    k[T+2] = k̄
    h = [hpath;h̄]
    #compute path of c given guess of $k$
    c = [k[t]^α*h[t]^(1-α) + (1-δ)*k[t] - k[t+1] for t in 1:T+1]
    y = [k[t]^α*h[t]^(1-α) for t in 1:T+1]
    return DataFrame(k=k[1:T],c=c[1:T],h=h[1:T],i=ipath,y=y[1:T])
end


"""
    construct_indirectU_nri!(model::RBCmodelNS)

Solves for the indirect utility and hours if entering with assets
kgrid[n] and saving kgrid[n′]
"""
function construct_indirectU_nri!(model::RBCmodelNS)
    @unpack α,β,δ,θ,kgrid = model
    N = length(kgrid)
    U = model.U = zeros(N,N)
    h = model.h = zeros(N,N)

    for n in 1:N
        for n′ in 1:N
            i = kgrid[n′] - (1-δ)*kgrid[n]
            k = kgrid[n]
            if i > k^α #output cannot cover investment
                U[n,n′] = -Inf
            elseif i <0 #cannot have 
                U[n,n′] = -Inf
            else
                #minimal labor supply to ensure positive consumption
                hmin = i>0 ? ((i+0.00001)/k^α)^(1/(1-α)) : 0.00001
                f(h) = ((1-α)*k^α*h^(-α)*θ/(k^α*h^(1-α)-i) - (1-θ)/(1-h)) #optimal labor leisure choice
                if f(hmin) <= 0 
                    #investment is negative enough that negative hours solves equation impose h=0
                    h[n,n′] = hmin
                else
                    h[n,n′] = brent(f,hmin,0.999999999999)
                end
                c = k^α*h[n,n′]^(1-α) -i
                U[n,n′] = θ*log(c) + (1-θ)*log(1-h[n,n′])
            end
        end 
    end
end



construct_indirectU_nri!(model)
V,n_pol,k_pol,h_pol = RBCsolve_bellman(model,zeros(length(model.kgrid)))

n0=400
df = simulate(model,n0,100,n_pol,h_pol)
df2 = solve_transition_nri(model,model.kgrid[n0],100)
plot(layer(df,y=:i,color=["bellman"]),layer(df2,y=:i,color=["Perfect Foresight"]))





@with_kw mutable struct RBCmodel
    β::Float64 
    α::Float64 
    θ::Float64 
    δ::Float64
    
    #Stochastic Properties
    A::Vector{Float64}
    P::Matrix{Float64}

    #steady state
    h̄::Float64 
    ī::Float64
    k̄::Float64
    c̄::Float64

    #objects for Bellman equation
    kgrid::Vector{Float64} = zeros(0)
    U::Array{Float64,3} = zeros(0,0,0)
    h::Array{Float64,3} = zeros(0,0,0)
end

"""
    RBCmodel()

Constructs RBCmodelNS based on calibration and stores steady state
"""
function RBCmodel(ρ,σ,N)
    β = 0.98
    α = 0.3
    #first compute steady state
    h̄ = 0.3
    δ = (1/β-1)/(α/0.16 - 1)
    k̄ = (δ/0.16*0.3^(α-1))^(1/(α-1))
    c̄ = k̄^α*h̄^(1-α)-δ*k̄
    θ = (1 + (1-h̄)*(1-α)*k̄^α*h̄^(-α)/c̄)^(-1)
    ī = δ*k̄
    mc = rouwenhorst(N,ρ,σ)
    return RBCmodel(β=β,α=α,θ=θ,δ=δ,h̄=h̄,ī=ī,k̄=k̄,c̄=c̄,A=exp.(mc.state_values),P=mc.p)
end


"""
    construct_indirectU!(model::RBCmodel)

Solves for the indirect utility and hours if entering with assets
kgrid[n] and saving kgrid[n′] if the aggregate state is s
"""
function construct_indirectU!(model::RBCmodel)
    @unpack α,β,δ,θ,kgrid,A = model
    N = length(kgrid)
    S = length(A)
    U = model.U = zeros(N,N,S)
    h = model.h = zeros(N,N,S)

    for s in 1:S
        for n in 1:N
            for n′ in 1:N
                i = kgrid[n′] - (1-δ)*kgrid[n]
                k = kgrid[n]
                if i + 0.000001 >= A[s]*k^α  #output cannot cover investment + minimal consumption
                    U[n,n′,s] = -Inf
                else
                    #minimal labor supply to ensure positive consumption
                    hmin = i>0 ? ((i+0.000001)/(A[s]*k^α))^(1/(1-α)) : 0.000001
                    f(h) = ((1-α)*A[s]*k^α*h^(-α)*θ/(A[s]*k^α*h^(1-α)-i) - (1-θ)/(1-h)) #optimal labor leisure choice
                    if f(hmin) <= 0 
                        #investment is negative enough that negative hours solves equation impose h=0
                        h[n,n′,s] = hmin
                    else
                        h[n,n′,s] = brent(f,hmin,0.999999999999)
                    end
                    c = A[s]*k^α*h[n,n′,s]^(1-α) -i
                    U[n,n′,s] = θ*log(c) + (1-θ)*log(1-h[n,n′,s])
                end
            end 
        end
    end
end


"""
    RBCbellmanmap(model::RBCmodelNS,V′)

Iterates on the bellman equation for the standard neoclassical growth model

# Arguments
* `V` Vector of values for each capital level in kgrid
"""
function RBCbellmanmap(model::RBCmodel,V′)
    @unpack kgrid,U,h,A,P,β = model
    N = length(kgrid)
    S = length(A)
    V = zeros(S,N) #new value function
    n_pol = zeros(Int,S,N) #policy rule for grid points
    k_pol = zeros(S,N) #policy rule for capital
    h_pol = zeros(S,N)
    obj = zeros(N)

    EV′ = P*V′
    for s in 1:S
        for n in 1:N #iterate for each initial capital
            obj .= U[n,:,s] .+ β.*EV′[s,:] #note using indirect utiltiy
            V[s,n],n_pol[s,n] = findmax(obj) #find optimal value and the choice that gives it
            k_pol[s,n] = kgrid[n_pol[s,n]] #record capital policy
            h_pol[s,n] = h[n,n_pol[s,n],s] #record hours policy
        end
    end
    return V,n_pol,k_pol,h_pol
end

"""
    RBCbellmanmap(model::RBCmodelNS,V′)

Iterates on the bellman equation for the standard neoclassical growth model

# Arguments
* `V` Vector of values for each capital level in kgrid
"""
function RBCbellmanmap_howard(model::RBCmodel,V′,n_pol)
    @unpack kgrid,U,h,P,β = model
    N = length(kgrid)
    S = size(P,1)
    V = zeros(S,N) #new value function
    EV′ = P*V′
    for s in 1:S
        for n in 1:N #iterate for each initial capital
            V[s,n] = U[n,n_pol[s,n],s] + β*EV′[s,n_pol[s,n]] #note using indirect utiltiy
        end
    end
    return V
end


"""
    simulate(model::RBCmodel,n_0,T,n_pol,h_pol)

Simulate the path of aggregates for T periods given initial state n_0
"""
function simulate(model::RBCmodel,n_0,s_0,T,n_pol,h_pol)
    @unpack α,β,δ,θ,kgrid,P,A = model
    k = zeros(T+1)
    n = zeros(Int,T+1)
    c,h,y,i = zeros(T),zeros(T),zeros(T),zeros(T)
    s = simulate_indices(MarkovChain(P),T,init=s_0)
    n[1] = n_0
    k[1] = kgrid[n_0]
    #no simulate using policies
    for t in 1:T
        n[t+1] = n_pol[s[t],n[t]]
        k[t+1] = kgrid[n[t+1]]
        h[t] = h_pol[s[t],n[t]]
        y[t] = A[s[t]]*k[t]^α*h[t]^(1-α)
        i[t] = k[t+1] - (1-δ)*k[t]
        c[t] = y[t] - i[t]
    end

    return DataFrame(k=k[1:T],h=h,c=c,i=i,y=y)
end



model = RBCmodel(0.779,0.007,25)
model.kgrid = LinRange(model.k̄*0.3,model.k̄*1.7,400)
construct_indirectU!(model)
V,n_pol,k_pol,h_pol = RBCsolve_bellman(model,zeros(25,400))

df = simulate(model,200,13,10000,n_pol,h_pol)
df = select!(df,Not(:k)) #drop k
df = last(df,7000) #drop first 3000
println(describe(df,:std,
                 (x->autocor(x,[1])[1])=>:autocorr,
                 (x->cor(x,df.h))=>:cor_h,
                 (x->cor(x,df.c))=>:cor_c,
                 (x->cor(x,df.i))=>:cor_i,
                 (x->cor(x,df.y))=>:cor_y,
                 ))

println(sum(df.i.<=0))


"""
    construct_indirectU_nri!(model::RBCmodel)

Solves for the indirect utility and hours if entering with assets
kgrid[n] and saving kgrid[n′] if the aggregate state is s. 
Investment is nonreversible
"""
function construct_indirectU_nri!(model::RBCmodel)
    @unpack α,β,δ,θ,kgrid,A = model
    N = length(kgrid)
    S = length(A)
    U = model.U = zeros(N,N,S)
    h = model.h = zeros(N,N,S)

    for s in 1:S
        for n in 1:N
            for n′ in 1:N
                i = kgrid[n′] - (1-δ)*kgrid[n]
                k = kgrid[n]
                if i + 0.000001 >= A[s]*k^α  #output cannot cover investment + minimal consumption
                    U[n,n′,s] = -Inf
                elseif i < 0
                    U[n,n′,s] = -Inf
                else
                    #minimal labor supply to ensure positive consumption
                    hmin = i>0 ? ((i+0.000001)/(A[s]*k^α))^(1/(1-α)) : 0.000001
                    f(h) = ((1-α)*A[s]*k^α*h^(-α)*θ/(A[s]*k^α*h^(1-α)-i) - (1-θ)/(1-h)) #optimal labor leisure choice
                    if f(hmin) <= 0 
                        #investment is negative enough that negative hours solves equation impose h=0
                        h[n,n′,s] = hmin
                    else
                        h[n,n′,s] = brent(f,hmin,0.999999999999)
                    end
                    c = A[s]*k^α*h[n,n′,s]^(1-α) -i
                    U[n,n′,s] = θ*log(c) + (1-θ)*log(1-h[n,n′,s])
                end
            end 
        end
    end
end

construct_indirectU_nri!(model)
V,n_pol,k_pol,h_pol = RBCsolve_bellman(model,zeros(25,400))

df = simulate(model,200,13,10000,n_pol,h_pol)
df = select!(df,Not(:k)) #drop k
df = last(df,7000) #drop first 3000
println(describe(df,:std,
                 (x->autocor(x,[1])[1])=>:autocorr,
                 (x->cor(x,df.h))=>:cor_h,
                 (x->cor(x,df.c))=>:cor_c,
                 (x->cor(x,df.i))=>:cor_i,
                 (x->cor(x,df.y))=>:cor_y,
                 ))