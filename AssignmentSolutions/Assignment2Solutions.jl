using Roots,QuantEcon,NLsolve,Parameters,Plots,DataFrames,LinearAlgebra,StatsPlots

@with_kw mutable struct RBCmodelNS
    β::Float64
    γ::Float64 
    α::Float64 
    χ::Float64 
    δ::Float64 
    ϕ::Float64

    #steady state
    n̄::Float64 
    ī::Float64
    k̄::Float64
    c̄::Float64

    #objects for Bellman equation
    kgrid::Vector{Float64} = zeros(0)
    U::Matrix{Float64} = zeros(0,0)
    n::Matrix{Float64} = zeros(0,0)
end

"""
    RBCmodelNS()

Constructs RBCmodelNS based on calibration and stores steady state
"""
function RBCmodelNS()
    β = 0.99
    α = 0.3
    γ = 2.
    #first compute steady state
    n̄ = 0.3
    δ = (1/β-1)/(α/0.16 - 1)
    k̄ = (δ/0.16*n̄^(α-1))^(1/(α-1))
    c̄ = k̄^α*n̄^(1-α)-δ*k̄
    χ = n̄^(-γ)*(1-α)*k̄^α*n̄^(-α)/c̄
    ī = δ*k̄
    return RBCmodelNS(β=β,α=α,χ=χ,γ=γ,δ=δ,ϕ=0.,n̄=n̄,ī=ī,k̄=k̄,c̄=c̄)
end;
model = RBCmodelNS()
println(model.χ)
println(model.δ)

"""
    path_residuals(model::RBCmodelNS,k0::Float64,ihpath)

compute residuals given a path of i and n
"""
function path_residuals!(model::RBCmodelNS,k0::Float64,inpath,T)
    @unpack α,β,δ,γ,χ,k̄,n̄,c̄ = model
    ipath,npath = inpath[1:T],inpath[T+1:end]
    k = zeros(T+2)
    k[1] = k0
    for t in 1:T
        k[t+1] = (1-δ)*k[t] + ipath[t] 
    end
    k[T+2] = k̄
    n = [npath;n̄]
    #compute path of c given guess of $k$
    c = [k[t]^α*n[t]^(1-α) + (1-δ)*k[t] - k[t+1] for t in 1:T+1]
    u_c = @. 1/c
    u_n = @. χ*n^γ
    F_n = @. (1-α)*k[1:T+1]^α*n^(-α)
    R = @. 1-δ+α*k[1:T+1]^(α-1)*n^(1-α)

    return [u_c[1:T].-β.*R[2:T+1].*u_c[2:T+1];
    u_c[1:T].*F_n[1:T].-u_n[1:T]]
end;

"""
    solve_transition(model::RBCmodelNS,k0,T)
    
returns a path of 
"""
function solve_transition(model::RBCmodelNS,k0,T)
    @unpack α,β,δ,k̄,n̄,c̄ = model
    f(x) = path_residuals!(model,k0,x,T)
    inpath0 = [model.ī*ones(T);model.n̄*ones(T)]

    res = nlsolve(f,inpath0)

    if !converged(res)
        error("Could not find root")
    end
    #Now back out aggregates
    inpath = res.zero
    ipath,npath = inpath[1:T],inpath[T+1:end]
    k = zeros(T+2)
    k[1] = k0
    for t in 1:T
        k[t+1] = (1-δ)*k[t] + ipath[t] 
    end
    k[T+2] = k̄
    n = [npath;n̄]
    #compute path of c given guess of $k$
    c = [k[t]^α*n[t]^(1-α) + (1-δ)*k[t] - k[t+1] for t in 1:T+1]
    y = [k[t]^α*n[t]^(1-α) for t in 1:T+1]
    return DataFrame(k=k[1:T],c=c[1:T],n=n[1:T],i=ipath,y=y[1:T])
end;

df = solve_transition(model,model.k̄+3,400)
@df df plot(:c,layout=(2,2),subplot=1,ylabel="Consumption")
@df df plot!(:n,subplot=2,ylabel="Labor")
@df df plot!(:k,subplot=3,ylabel="Capital")
@df df plot!(:i,subplot=4,ylabel="Investment")

df = solve_transition(model,model.k̄-3,400)
@df df plot(:c,layout=(2,2),subplot=1,ylabel="Consumption")
@df df plot!(:n,subplot=2,ylabel="Labor")
@df df plot!(:k,subplot=3,ylabel="Capital")
@df df plot!(:i,subplot=4,ylabel="Investment")

model.kgrid = LinRange(model.k̄*0.8,model.k̄*1.2,1000)

"""
    construct_indirectU!(model::RBCmodelNS)

Solves for the indirect utility and hours if entering with assets
kgrid[n] and saving kgrid[n′]
"""
function construct_indirectU!(model::RBCmodelNS)
    @unpack α,β,δ,χ,γ,kgrid = model
    N = length(kgrid)
    U = model.U = zeros(N,N)
    n = model.n = zeros(N,N)

    for j in 1:N
        for j′ in 1:N
            i = kgrid[j′] - (1-δ)*kgrid[j]
            k = kgrid[j]

            #minimal labor supply to ensure positive consumption
            nmin = i>0 ? ((i+0.00001)/k^α)^(1/(1-α)) : 0.00001
            f(n) = (1-α)*k^α*n^(-α)/(k^α*n^(1-α)-i) - χ*n^γ #optimal labor leisure choice
            n[j,j′] = brent(f,nmin,100.)
            
            c = k^α*n[j,j′]^(1-α) -i
            U[j,j′] = log(c) - χ*n[j,j′]^(1+γ)/(1+γ)
        end 
    end
end
construct_indirectU!(model);

"""
    RBCbellmanmap(model::RBCmodelNS,V′)

Iterates on the bellman equation for the standard neoclassical growth model

# Arguments
* `V` Vector of values for each capital level in kgrid
"""
function RBCbellmanmap(model::RBCmodelNS,V′)
    @unpack kgrid,U,n,β = model
    N = length(kgrid)
    V = zeros(N) #new value function
    j_pol = zeros(Int,N) #policy rule for grid points
    k_pol = zeros(N) #policy rule for capital
    n_pol = zeros(N)
    obj = zeros(N)
    for j in 1:N #iterate for each initial capital
        obj .= U[j,:] .+ β.*V′ #note using indirect utiltiy
        V[j],j_pol[j] = findmax(obj) #find optimal value and the choice that gives it
        k_pol[j] = kgrid[j_pol[j]] #record capital policy
        n_pol[j] = n[j,j_pol[j]] #record hours policy
    end
    return V,j_pol,k_pol,n_pol
end;

"""
RBCbellmanmap_howard(model::RBCmodelNS,V′,j_pol)

Iterates on the bellman equation for the standard neoclassical growth model

# Arguments
* `V` Vector of values for each capital level in kgrid
* j_pol capital policy
"""
function RBCbellmanmap_howard(model::RBCmodelNS,V′,j_pol)
    @unpack kgrid,U,n,β = model
    N = length(kgrid)
    V = zeros(N) #new value function
    for j in 1:N #iterate for each initial capital
        V[j] = U[j,j_pol[j]] + β*V′[j_pol[j]] #note using indirect utiltiy
    end
    return V
end;

"""
    RBCsolve_bellman(model,V0,[,ϵ=1e-8])

Solves the bellman equation by iterating until convergence

# Arguments
* `V0` Initial vector of values for each capital level in kgrid
* `ϵ` Convergence criteria
* H=100 seems to be close to optimal for this problem
"""
function RBCsolve_bellman(model,V0,H=100,ϵ=1e-8)
    diff = 1.
    V,j_pol,k_pol,n_pol = RBCbellmanmap(model,V0)
    while diff > ϵ
        for h in 1:H
            V= RBCbellmanmap_howard(model,V,j_pol)
        end
        V_new,j_pol,k_pol,n_pol = RBCbellmanmap(model,V)
        diff = norm(V_new-V,Inf)
        println(diff)
        V = V_new 
    end
    return V,j_pol,k_pol,n_pol
end
V,j_pol,k_pol,n_pol = RBCsolve_bellman(model,zeros(length(model.kgrid)));

"""
    simulate(model::RBCmodelNS,j_0,T,n_pol,h_pol)

Simulate the path of aggregates for T periods given initial state j_0
"""
function simulate(model::RBCmodelNS,j_0,T,j_pol,n_pol)
    @unpack α,β,δ,χ,γ,kgrid = model
    k = zeros(T+1)
    j = zeros(Int,T+1)
    c,n,y,i = zeros(T),zeros(T),zeros(T),zeros(T)
    j[1] = j_0
    k[1] = kgrid[j_0]
    #no simulate using policies
    for t in 1:T
        j[t+1] = j_pol[j[t]]
        k[t+1] = kgrid[j[t+1]]
        n[t] = n_pol[j[t]]
        y[t] = k[t]^α*n[t]^(1-α)
        i[t] = k[t+1] - (1-δ)*k[t]
        c[t] = y[t] - i[t]
    end

    return DataFrame(k=k[1:T],n=n,c=c,i=i,y=y)
end;

j0=50
df = simulate(model,j0,400,j_pol,n_pol)
df2 = solve_transition(model,model.kgrid[j0],400)
plot(df.k,label="Bellman")
plot!(df2.k,label="Perfect Foresight")

j0=350
df = simulate(model,j0,100,j_pol,n_pol)
df2 = solve_transition(model,model.kgrid[j0],100)
plot(df.k,label="Bellman")
plot!(df2.k,label="Perfect Foresight")

"""
    path_residuals_ϕ(model::RBCmodelNS,k0::Float64,ihpath)

compute residuals given a path of i and n.  Allows for ϕ>0
"""
function path_residuals_ϕ!(model::RBCmodelNS,k0::Float64,res,inpath,T)
    @unpack α,β,δ,γ,χ,ϕ,k̄,n̄,c̄,ī = model
    ipath,npath = inpath[1:T],inpath[T+1:end]
    k = zeros(T+2)
    k[1] = k0
    for t in 1:T
        k[t+1] = (1-δ)*k[t] + ipath[t] 
    end
    i = [ipath;ī]
    k[T+2] = k̄
    n = [npath;n̄]
    #compute path of c given guess of $k$
    c = [k[t]^α*n[t]^(1-α) + (1-δ)*k[t] - k[t+1] - ϕ*(i[t]/k[t]-δ)^2*k[t] for t in 1:T+1]
    u_c = @. 1/c
    u_n = @. χ*n^γ
    F_n = @. (1-α)*k[1:T+1]^α*n^(-α)
    F_k = @. α*k[1:T+1]^(α-1)*n^(1-α)
    q   = @. 1+2*ϕ*(i/k[1:T+1]-δ)

    #Rk = @. F_k*k[1:T+1] - ϕ*(i/k[1:T+1]-δ)^2*k[1:T+1] - i + q*k[2:T+2] #@. 1-δ+α*k[1:T+1]^(α-1)*n^(1-α)
    D = @. F_k*k[1:T+1] - ϕ*(i/k[1:T+1]-δ)^2*k[1:T+1] - i 

    @. res[1:T] = u_c[1:T]*q[1:T]*k[2:T+1]-β*u_c[2:T+1]*(D[2:T+1]+q[2:T+1]*k[3:T+2])
    res[T+1:end] .= u_c[1:T].*F_n[1:T].-u_n[1:T]
end;

"""
    solve_transition_ϕ(model::RBCmodelNS,k0,T)
    
returns a path of 
"""
function solve_transition_ϕ(model::RBCmodelNS,k0,T)
    @unpack α,β,ϕ,δ,k̄,n̄,c̄,ī = model
    f!(F,x) = path_residuals_ϕ!(model,k0,F,x,T)
    inpath0 = [model.ī*ones(T);model.n̄*ones(T)]

    res = nlsolve(f!,inpath0)

    if !converged(res)
        error("Could not find root")
    end
    #Now back out aggregates
    inpath = res.zero
    ipath,npath = inpath[1:T],inpath[T+1:end]
    k = zeros(T+2)
    k[1] = k0
    for t in 1:T
        k[t+1] = (1-δ)*k[t] + ipath[t] 
    end
    k[T+2] = k̄
    n = [npath;n̄]
    i = [ipath;ī]
    #compute path of c given guess of $k$
    c = [k[t]^α*n[t]^(1-α) + (1-δ)*k[t] - k[t+1] - ϕ*(i[t]/k[t]-δ)^2*k[t] for t in 1:T+1]
    y = [k[t]^α*n[t]^(1-α) for t in 1:T+1]
    return DataFrame(k=k[1:T],c=c[1:T],n=n[1:T],i=i[1:T],y=y[1:T])
end;

model.ϕ = 10.
df = solve_transition(model,model.k̄+3,400) #solves transition assuming ϕ=0
dfϕ = solve_transition_ϕ(model,model.k̄+3,400)
@df df plot(:c,layout=(2,2),subplot=1,label="ϕ=0",ylabel="Consumption")
@df dfϕ plot!(:c,subplot=1,label="ϕ=10",ylabel="Consumption")
@df df plot!(:n,subplot=2,ylabel="Labor")
@df dfϕ plot!(:n,subplot=2,legend=false)
@df df plot!(:k,subplot=3,ylabel="Capital")
@df dfϕ plot!(:k,subplot=3,legend=false)
@df df plot!(:i,subplot=4,ylabel="Investment")
@df dfϕ plot!(:i,subplot=4,legend=false)

dfϕ.q = 1 .+ 2*model.ϕ*(dfϕ.i./dfϕ.k.-model.δ)

@df dfϕ plot(:q)

"""
    construct_indirectU_ϕ!(model::RBCmodelNS)

Solves for the indirect utility and hours if entering with assets
kgrid[n] and saving kgrid[n′]
"""
function construct_indirectU_ϕ!(model::RBCmodelNS)
    @unpack α,β,δ,χ,γ,ϕ,kgrid = model
    N = length(kgrid)
    U = model.U = zeros(N,N)
    n = model.n = zeros(N,N)

    for j in 1:N
        for j′ in 1:N
            i = kgrid[j′] - (1-δ)*kgrid[j]
            k = kgrid[j]
            ϕcost = ϕ*(i/k-δ)^2*k 
            #minimal labor supply to ensure positive consumption
            nmin = i+ϕcost>0 ? ((i+0.000001+ϕcost)/k^α)^(1/(1-α)) : 0.
            f(n) = (1-α)*k^α*n^(-α)/(k^α*n^(1-α)-i -ϕcost) - χ*n^γ #optimal labor leisure choice
            n[j,j′] = brent(f,nmin,100.)
            
            c = k^α*n[j,j′]^(1-α) -i-ϕcost
            U[j,j′] = log(c) - χ*n[j,j′]^(1+γ)/(1+γ)
        end 
    end
end
construct_indirectU_ϕ!(model);

V,j_pol,k_pol,n_pol = RBCsolve_bellman(model,zeros(length(model.kgrid)));

n0= 400
df = simulate(model,n0,400,j_pol,n_pol)
df2 = solve_transition_ϕ(model,model.kgrid[n0],400)
plot(df.k,label="Bellman")
plot!(df2.k,label="Perfect Foresight")

@with_kw mutable struct RBCmodel
    β::Float64
    γ::Float64 
    α::Float64 
    χ::Float64 
    δ::Float64 
    ϕ::Float64


    #Stochastic Properties
    A::Vector{Float64}
    P::Matrix{Float64}

    #steady state
    n̄::Float64 
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
    β = 0.99
    α = 0.3
    γ = 2.
    #first compute steady state
    n̄ = 0.3
    δ = (1/β-1)/(α/0.16 - 1)
    k̄ = (δ/0.16*n̄^(α-1))^(1/(α-1))
    c̄ = k̄^α*n̄^(1-α)-δ*k̄
    χ = n̄^(-γ)*(1-α)*k̄^α*n̄^(-α)/c̄
    ī = δ*k̄
    mc = rouwenhorst(N,ρ,σ)
    return RBCmodel(β=β,α=α,χ=χ,γ=γ,δ=δ,ϕ=0.,n̄=n̄,ī=ī,k̄=k̄,c̄=c̄,A=exp.(mc.state_values),P=mc.p)
end;
model = RBCmodel(0.779,0.007,10)

model.kgrid = LinRange(model.k̄*0.8,model.k̄*1.2,400);

"""
    construct_indirectU!(model::RBCmodel)

Solves for the indirect utility and hours if entering with assets
kgrid[n] and saving kgrid[n′] if the aggregate state is s
"""
function construct_indirectU!(model::RBCmodel)
    @unpack α,β,δ,χ,γ,ϕ,kgrid,A = model
    N = length(kgrid)
    S = length(A)
    U = model.U = zeros(N,N,S)
    n = model.h = zeros(N,N,S)

    for s in 1:S
        for j in 1:N
            for j′ in 1:N
                i = kgrid[j′] - (1-δ)*kgrid[j]
                k = kgrid[j]
                ϕcost = ϕ*(i/k-δ)^2*k 
                #minimal labor supply to ensure positive consumption
                nmin = i+ϕcost>0 ? ((i+0.000001+ϕcost)/(A[s]*k^α))^(1/(1-α)) : 0.
                f(n) = (1-α)*A[s]*k^α*n^(-α)/(A[s]*k^α*n^(1-α)-i -ϕcost) - χ*n^γ #optimal labor leisure choice
                n[j,j′,s] = brent(f,nmin,100.)
                
                c = A[s]*k^α*n[j,j′,s]^(1-α) -i-ϕcost
                U[j,j′,s] = log(c) - χ*n[j,j′,s]^(1+γ)/(1+γ)
            end 
        end
    end
end;
construct_indirectU!(model);

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
    j_pol = zeros(Int,S,N) #policy rule for grid points
    k_pol = zeros(S,N) #policy rule for capital
    n_pol = zeros(S,N)
    obj = zeros(N)

    EV′ = P*V′
    for s in 1:S
        for j in 1:N #iterate for each initial capital
            obj .= U[j,:,s] .+ β.*EV′[s,:] #note using indirect utiltiy
            V[s,j],j_pol[s,j] = findmax(obj) #find optimal value and the choice that gives it
            k_pol[s,j] = kgrid[j_pol[s,j]] #record capital policy
            n_pol[s,j] = h[j,j_pol[s,j],s] #record hours policy
        end
    end
    return V,j_pol,k_pol,n_pol
end

"""
    RBCbellmanmap_howard(model::RBCmodelNS,V′)

Iterates on the bellman equation for the standard neoclassical growth model
using the howard improvement algorithm

# Arguments
* `V′` Vector of values for each capital level in kgrid
"""
function RBCbellmanmap_howard(model::RBCmodel,V′,j_pol)
    @unpack kgrid,U,h,P,β = model
    N = length(kgrid)
    S = size(P,1)
    V = zeros(S,N) #new value function
    EV′ = P*V′
    for s in 1:S
        for j in 1:N #iterate for each initial capital
            V[s,j] = U[j,j_pol[s,j],s] + β*EV′[s,j_pol[s,j]] #note using indirect utiltiy
        end
    end
    return V
end
V,j_pol,k_pol,n_pol = RBCsolve_bellman(model,zeros(10,400));

"""
    simulate(model::RBCmodel,j_0,s_0,T,j_pol,n_pol)

Simulate the path of aggregates for T periods given initial state j_0,s_0
"""
function simulate(model::RBCmodel,j_0,s_0,T,j_pol,n_pol)
    @unpack α,β,δ,ϕ,kgrid,P,A = model
    k = zeros(T+1)
    j = zeros(Int,T+1)
    c,n,y,i = zeros(T),zeros(T),zeros(T),zeros(T)
    s = simulate_indices(MarkovChain(P),T,init=s_0)
    j[1] = j_0
    k[1] = kgrid[j_0]
    #no simulate using policies
    for t in 1:T
        j[t+1] = j_pol[s[t],j[t]]
        k[t+1] = kgrid[j[t+1]]
        n[t] = n_pol[s[t],j[t]]
        y[t] = A[s[t]]*k[t]^α*n[t]^(1-α)
        i[t] = k[t+1] - (1-δ)*k[t]
        c[t] = y[t] - i[t] - ϕ*(i[t]/k[t]-δ)^2*k[t]
    end

    return DataFrame(k=k[1:T],n=n,c=c,i=i,y=y)
end
df = simulate(model,200,5,10000,j_pol,n_pol);

using Statistics
df = select!(df,Not(:k)) #drop k
df = last(df,7000) #drop first 3000
describe(df,:std,
                 (x->autocor(x,[1])[1])=>:autocorr,
                 (x->cor(x,df.n))=>:cor_n,
                 (x->cor(x,df.c))=>:cor_c,
                 (x->cor(x,df.i))=>:cor_i,
                 (x->cor(x,df.y))=>:cor_y,
                 )

model.ϕ = 10.
construct_indirectU!(model);

V,n_pol,k_pol,h_pol = RBCsolve_bellman(model,zeros(10,400))

df = simulate(model,200,5,10000,n_pol,h_pol)
df = select!(df,Not(:k)) #drop k
df = last(df,7000) #drop first 3000
describe(df,:std,
                 (x->autocor(x,[1])[1])=>:autocorr,
                 (x->cor(x,df.n))=>:cor_n,
                 (x->cor(x,df.c))=>:cor_c,
                 (x->cor(x,df.i))=>:cor_i,
                 (x->cor(x,df.y))=>:cor_y,
                 )