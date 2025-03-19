using BasisMatrices,LinearAlgebra,Parameters,Optim,QuantEcon,DataFrames,StatsPlots,SparseArrays,Arpack,Roots
default(linewidth=2,legend=false)
@with_kw mutable struct HHModel
    #Preference Parameters
    γ::Float64 = 1. #Risk aversion
    β::Float64 = 0.985 #Discount Rate

    #Prices
    r̄::Float64 = .01 #quarterly
    w̄::Float64 = 1.

    #Asset Grid Parameters
    a̲::Float64 = 0. #Borrowing Constraint
    a̅::Float64 = 600. #Upper Bound on assets
    Na::Int64 = 100 #Number of grid points for splines

    #Income Process
    ρ_ϵ::Float64 = 0.9923 #calibrated to quarterly wage regressions
    σ_ϵ::Float64 = 0.0983
    Nϵ::Int64 = 7
    ϵ::Vector{Float64} = zeros(0)
    Π::Matrix{Float64} = zeros(0,0)

    #Solution
    k::Int = 2 #type of interpolation
    Vf::Vector{Interpoland} = Interpoland[]
    cf::Vector{Interpoland} = Interpoland[]

    #Extra
    EΦ′::SparseMatrixCSC{Float64,Int64} = spzeros(0,0)
    Φ::SparseMatrixCSC{Float64,Int64} = spzeros(0,0)
end;

"""
    U(HH::HHModel,c)
"""
function U(HH::HHModel,c)
    γ = HH.γ
    if γ == 1
        return log.(c)
    else
        return (c.^(1-γ))./(1-γ)
    end
end

"""
    setupgrids_shocks!(HH::HHModel, curv=1.7)

Set up non-linear grids for interpolation
"""
function setupgrids_shocks!(HH::HHModel, curv=1.7)
    @unpack a̲,a̅,Na,ρ_ϵ,σ_ϵ,Nϵ,k,r̄,w̄,β = HH
    #Compute grid on A
    agrid = (a̅-a̲).*LinRange(0,1,Na).^curv .+ a̲

    #Store markov chain
    mc = rouwenhorst(Nϵ,ρ_ϵ,σ_ϵ)
    HH.Π = Π = mc.p
    HH.ϵ = exp.(mc.state_values)

    #First guess of interpolation functions
    abasis = Basis(SplineParams(agrid,0,k))
    a = nodes(abasis)[1]

    Vf = HH.Vf = Vector{Interpoland}(undef,Nϵ)
    cf = HH.cf = Vector{Interpoland}(undef,Nϵ)
    for s in 1:Nϵ
        c = @. r̄*a + w̄*HH.ϵ[s]
        V = U(HH,c)./(1-β)

        Vf[s]= Interpoland(abasis,V)
        cf[s]= Interpoland(abasis,c)
    end
end;

"""
    iterate_endogenousgrid(HH,a′grid,cf′)

Iterates on Euler equation using endogenous grid method
"""
function iterate_endogenousgrid(HH,a′grid,cf′)
    @unpack γ,ϵ,β,Nϵ,Π,r̄,w̄,a̲= HH
    c′ = zeros(length(a′grid),Nϵ)
    for s in 1:Nϵ
        c′[:,s]= cf′[s](a′grid)
    end

    EERHS = β*(1+r̄)*(c′).^(-γ)*Π' #RHS of Euler Equation
    c = EERHS.^(-1/γ)

    #compute implies assets
    a = ((c .+ a′grid) .- w̄ .*ϵ')./(1+r̄)

    cf = Vector{Interpoland}(undef,Nϵ)
    for s in 1:Nϵ
        if a[1,s]> a̲
            c̲ = r̄*a̲ + w̄*ϵ[s]
            cf[s]= Interpoland(Basis(SplineParams([a̲; a[:,s]],0,1)),[c̲;c[:,s]])
        else
            cf[s]= Interpoland(Basis(SplineParams(a[:,s],0,1)),c[:,s])
        end
    end
    return cf
end;

"""
    solveHHproblem_eg!(HH)

Solves the HH problem using the endogeneous grid method
"""
function solveHHproblem_eg!(HH,verbose=false)
    a′grid = nodes(HH.Vf[1].basis)[1]#Get nodes for interpolation
    
    cf′ = iterate_endogenousgrid(HH,a′grid,HH.cf)
    diff = 1.
    while diff  > 1e-8
        HH.cf = iterate_endogenousgrid(HH,a′grid,cf′)
        diff = maximum(norm(cf′[s](a′grid)-HH.cf[s](a′grid),Inf) for s in 1:HH.Nϵ) 
        if verbose
            println(diff)
        end
        cf′ = HH.cf
    end
end
HH = HHModel()
setupgrids_shocks!(HH,3.)
solveHHproblem_eg!(HH)

@with_kw mutable struct AiyagariModel
    HH::HHModel = HHModel()

    #Production Parameters
    α::Float64 = 0.3
    δ::Float64 = 0.025
    Θ̄::Float64 = 1.
    ρ_Θ::Float64 = 0.85

    #Moments to match/prices
    W̄::Float64 = 1.
    R̄::Float64 = 1.01
    K2Y::Float64 = 10.2 #capital to output ratio
    N̄::Float64 = 1.
    K̄::Float64 =1.

    #Distribution Parameters
    Ia::Int = 1000 #Number of gridpoints for distribution
    z̄::Matrix{Float64} = zeros(0,0) #Gridpoints for the state variables
    ω̄::Vector{Float64} = zeros(0) #Fraction of agents at each grid level
    H::SparseMatrixCSC{Float64,Int64} = spzeros(Ia,Ia) #Transition matrix
end;

"""
    setupgrids_shocks!(AM::AiyagariModel)

Setup the grids and shocks for the aiyagari model
"""
function setupgrids_shocks!(AM::AiyagariModel,curv=2.)
    @unpack HH,Ia,N̄= AM
    @unpack a̲,a̅,Nϵ = HH
    setupgrids_shocks!(HH)
    #Normalize so that average labor supply is 1
    πstat = real(eigs(HH.Π',nev=1)[2])
    πstat ./= sum(πstat)
    HH.ϵ = HH.ϵ./dot(πstat,HH.ϵ)*N̄
    #Grid for distribution
    agrid = (a̅-a̲).*LinRange(0,1,Ia).^curv .+ a̲
    AM.z̄ = hcat(kron(ones(Nϵ),agrid),kron(1:Nϵ,ones(Ia)))
    AM.ω̄ = ones(Ia*Nϵ)/(Ia*Nϵ)
end;

"""
    find_stationarydistribution!(AM::AiyagariModel,V)

Computes the stationary distribution 
"""
function find_stationarydistribution!(AM::AiyagariModel)
    @unpack Ia,z̄,HH,W̄,R̄ = AM
    @unpack ϵ,Π,Nϵ,cf,a̲,a̅ = HH

    a = z̄[1:Ia,1] #grids are all the same for all shocks
    c = hcat([cf[s](a) for s in 1:Nϵ]...) #consumption policy IaxNϵ
    a′ = R̄.*a .+ W̄.*ϵ' .- c #create a IaxNϵ grid for the policy rules
    
    #make sure we don't go beyond bounds.  Shouldn't bind if bmax is correct
    a′ = max.(min.(a′,a̅),a̲)
    
    Qa = BasisMatrix(Basis(SplineParams(a,0,1)),Direct(),reshape(a′,Ia*Nϵ)).vals[1]
    Q = spzeros(Ia*Nϵ,Ia*Nϵ)
    for s in 1:Nϵ
        Q[1+(s-1)*Ia:s*Ia,:] = kron(reshape(Π[s,:],1,:),Qa[1+(s-1)*Ia:s*Ia,:]) 
    end
    
    AM.H = Q'
    AM.ω̄ .= real(eigs(AM.H;nev=1)[2])[:]
    AM.ω̄ ./= sum(AM.ω̄) #normalize eigenvector
end;

function calibratesteadystate!(AM)
    @unpack Θ̄,α,N̄,K2Y,R̄ = AM
    AM.HH.r̄ = R̄ - 1
    Y2K = 1/K2Y
    AM.δ = α*Y2K + 1 - R̄ #matching capital to output ratio and interest rate gives depreciation rate
    K2N = (Y2K/Θ̄)^(1/(α-1)) #relationship between capital to output and capital to labor
    K̄ = AM.K̄ = K2N*N̄
    AM.W̄ = AM.HH.w̄ = (1-α)*Θ̄*K2N^α

    setupgrids_shocks!(AM)
    function βres(β)
        AM.HH.β=β
        solveHHproblem_eg!(AM.HH)
        find_stationarydistribution!(AM)
        return dot(AM.ω̄,AM.z̄[:,1]) -K̄
    end

    Q̄ = 1/R̄
    AM.HH.β =fzero(βres,Q̄^2,Q̄^1.2)
    solveHHproblem_eg!(AM.HH)
    find_stationarydistribution!(AM)
    return AM.HH.β
end

AM = AiyagariModel()
AM.R̄ = 1.01 #target a quarterly interest rate of 1%
AM.HH.β = 0.99
setupgrids_shocks!(AM)
calibratesteadystate!(AM)

"""
    iterate_endogenousgrid_transition(HH,a′grid,cf′,r,w,r′)

Iterates on Euler equation using endogenous grid method
"""
function iterate_endogenousgrid_transition(HH,a′grid,cf′,r,w,r′)
    @unpack γ,ϵ,β,Nϵ,Π,a̲= HH
    c′ = zeros(length(a′grid),Nϵ)
    for s in 1:Nϵ
        c′[:,s]= cf′[s](a′grid)
    end

    EERHS = β*(1+r′)*(c′).^(-γ)*Π' #RHS of Euler Equation
    c = EERHS.^(-1/γ)

    #compute implies assets
    a = ((c .+ a′grid) .- w .*ϵ')./(1+r)

    cf = Vector{Interpoland}(undef,Nϵ)
    for s in 1:Nϵ
        if a[1,s]> a̲
            c̲ = r*a̲ + w*ϵ[s]
            cf[s]= Interpoland(Basis(SplineParams([a̲; a[:,s]],0,1)),[c̲;c[:,s]])
        else
            cf[s]= Interpoland(Basis(SplineParams(a[:,s],0,1)),c[:,s])
        end
    end
    return cf
end;

"""
    compute_consumption_path(AM,rt,wt)

Computes path of prices given 
"""
function compute_consumption_path(AM,rt,wt)
    HH = AM.HH
    T = length(rt)
    cft = Matrix{Interpoland}(undef,AM.HH.Nϵ,T)
    #First compute last period using steady state as continuation
    a′grid = nodes(HH.Vf[1].basis)[1]#Get nodes for interpolation
    cft[:,T] =  iterate_endogenousgrid_transition(HH,a′grid,HH.cf,rt[T],wt[T],HH.r̄)
    
    #Next compute consumption in all previous periods working backwards
    for t in reverse(1:T-1)
        cft[:,t] .= iterate_endogenousgrid_transition(HH,a′grid,cft[:,t+1],rt[t],wt[t],rt[t+1])
    end

    return cft
end

"""
    iterate_distribution(AM::AiyagariModel,cf,λ,r,w)

Computes next periods distribution given current consumption function
"""
function iterate_distribution(AM::AiyagariModel,cf,ω,r,w)
    @unpack Ia,z̄,HH = AM
    @unpack ϵ,Π,Nϵ,a̲,a̅ = HH

    a = z̄[1:Ia,1] #grids are all the same for all shocks
    c = hcat([cf[s](a) for s in 1:Nϵ]...) #consumption policy IaxNϵ
    a′ = (1+r)*a .+ w.*ϵ' .- c #create a IaxNϵ grid for the policy rules
    
    #make sure we don't go beyond bounds.  Shouldn't bind if bmax is correct
    a′ = max.(min.(a′,a̅),a̲)
    
    Qa = BasisMatrix(Basis(SplineParams(a,0,1)),Direct(),reshape(a′,Ia*Nϵ)).vals[1]
    Q = spzeros(Ia*Nϵ,Ia*Nϵ)
    for s in 1:Nϵ
        Q[1+(s-1)*Ia:s*Ia,:] = kron(reshape(Π[s,:],1,:),Qa[1+(s-1)*Ia:s*Ia,:]) 
    end
    
    return Q'*ω
end;


"""
    compute_capital_path(AM,rt,wt)

Computes path of prices given 
"""
function compute_capital_path(AM,cft,rt,wt)
    @unpack HH,z̄,ω̄ = AM
    T = length(rt)
    N = length(AM.ω̄)
    ωt = zeros(N,T+1)
    ωt[:,1] = AM.ω̄ #starting at steady state distribution
    for t in 1:T
        ωt[:,t+1] = iterate_distribution(AM,cft[:,t],ωt[:,t],rt[t],wt[t])
    end
   
    return (z̄[:,1]'*ωt)'
end

"""
    solve_transition(AM,lnΘ0=0.05,T=200)

Solves for the perfect foresight equilibrium
"""
function solve_transition(AM,lnΘ0=0.05,T=200)
    @unpack α,δ = AM
    Θt = exp.([lnΘ0*AM.ρ_Θ^(t-1) for t in 1:T])
    Kt = ones(T).*AM.K̄ #initialize guess as constant capital
    
    diff = 1
    ξ = 0.1
    while diff > 1e-4
        rt = α.*Θt.*Kt.^(α-1) .- δ
        wt = (1-α).*Θt.*Kt.^(α)

        cft = compute_consumption_path(AM,rt,wt)
        K̂t = compute_capital_path(AM,cft,rt,wt)
        diff = norm(Kt-K̂t[1:T],Inf)
        println(diff)
        Kt = (1-ξ).*Kt .+ ξ.*K̂t[1:T]
    end
    rt = α.*Θt.*Kt.^(α-1) .- δ
    wt = (1-α).*Θt.*Kt.^(α)
    return Θt,Kt,rt,wt
end

Θt,Kt,rt,wt = solve_transition(AM,0.05,200);

plot(Kt,layout=4,subplot=1,title="Capital")
plot!(rt,subplot=2,title="Rental Rate")
plot!(wt,subplot=3,title="Wage")
plot!(Θt,subplot=4,title="Productivity")

ϵ = 1e-4
Θt,Kt,rt,wt = solve_transition(AM,ϵ,200);
dΘ̄ = (Θt.-1)./ϵ
dK̄ = (Kt.-AM.K̄)./ϵ
dr̄ = (rt.-AM.HH.r̄)./ϵ
dw̄ = (wt.-AM.HH.w̄)./ϵ 

T = 1000
σ_Θ = 0.015
Tlin = length(dK̄)
Θt,Kt,rt,wt = zeros(T+Tlin),zeros(T+Tlin),zeros(T+Tlin),zeros(T+Tlin)

#now simulate
for t in 1:T
    ε = randn()*σ_Θ
    Θt[t:t+Tlin-1] .+= dΘ̄.*ε
    Kt[t:t+Tlin-1] .+= dK̄.*ε
    rt[t:t+Tlin-1] .+= dr̄.*ε
    wt[t:t+Tlin-1] .+= dw̄.*ε
end
plot(Kt[1:T],layout=4,subplot=1,title="Capital")
plot!(rt[1:T],subplot=2,title="Rental Rate")
plot!(wt[1:T],subplot=3,title="Wage")
plot!(Θt[1:T],subplot=4,title="Productivity")

Θt,Kt,rt,wt = solve_transition(AM,σ_Θ,200);

plot(Kt,label="Non-Linear")
plot!(dK̄.*σ_Θ.+AM.K̄,label="Linear",legend=true)