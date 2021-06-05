using BasisMatrices,LinearAlgebra,Parameters,Optim,QuantEcon,DataFrames,Gadfly,SparseArrays,Arpack
using Roots

@with_kw mutable struct HHModel
    #Preference Parameters
    γ::Float64 = 1. #Risk aversion
    β::Float64 = 0.985 #Discount Rate

    #Prices
    r̄::Float64 = .01
    w̄::Float64 = 1.

    #Asset Grid Parameters
    a̲::Float64 = 0. #Borrowing Constraint
    a̅::Float64 = 400. #Upper Bound on assets
    Na::Int64 = 100.

    #Income Process
    ρ_ϵ::Float64 = 0.9923
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
end

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

    #Expectations of 1st derivative of Basis functions
    HH.EΦ′ = kron(Π,BasisMatrix(abasis,Direct(),nodes(abasis)[1],[1]).vals[1])
    HH.Φ = kron(Matrix{Float64}(I,Nϵ,Nϵ),BasisMatrix(abasis,Direct()).vals[1])
end


"""
    optimalPolicy(HH,Vf′)

Computes the  optimalPolicy given value function Vf′ if the state is (a,s)
"""
function optimalPolicy(HH,a,s,Vf′)
    @unpack a̲,a̅,β,Π,ϵ,r̄,w̄,Nϵ = HH 

    function objf(a′)
        c = (1+r̄)*a + ϵ[s]*w̄ - a′
        return U(HH,c) + β*sum(Π[s,s′]*Vf′[s′](a′) for s′ in 1:Nϵ)
    end
    a_max = min((1+r̄)*a+ϵ[s]*w̄,a̅)
    res = maximize(objf,a̲,a_max)
    a′ = Optim.maximizer(res)
    #return value and consumption that optimize
    return (V=objf(a′),c=(1+r̄)*a + ϵ[s]*w̄ - a′)
end

"""
    iterateBellman(HH,Vf′)

Iterates the Bellman equation given value function Vf′
"""
function iterateBellman(HH,Vf′)
    Nϵ = length(Vf′)
    Vf = Vector{Interpoland}(undef,length(Vf′))

    for s in 1:Nϵ
        basis = Vf′[s].basis
        V = [optimalPolicy(HH,a,s,Vf′).V for a in nodes(basis)[1]]
        Vf[s]= Interpoland(basis,V)
    end

    return Vf
end

"""
    solveBellman!(HH)
Solves the Bellman Equation for the HH problem
"""
function solveBellman!(HH)
    Nϵ = HH.Nϵ
    diff = 1.
    Vf′ = copy(HH.Vf)
    while diff > 1e-8 
        HH.Vf = iterateBellman(HH,Vf′)
        diff = maximum([norm(Vf′[s].coefs - HH.Vf[s].coefs,Inf) for s in 1:Nϵ])
        println(diff)
        Vf′ = HH.Vf
    end
end

HH = HHModel()
setupgrids_shocks!(HH)
iterateBellman(HH,HH.Vf)

@profview iterateBellman(HH,HH.Vf)




"""
computeoptimalconsumption(HH::HHModel,V)

Computes optimal savings using endogenous grid method.  
"""
function computeoptimalconsumption(HH::HHModel,Vcoefs::Vector{Float64})::Vector{Interpoland}
    @unpack γ,β,ϵ,Nϵ,a̲,EΦ′,r̄,w̄ = HH
    a′grid = nodes(HH.Vf[1].basis)[1]
    EV_a = reshape(EΦ′*Vcoefs,:,Nϵ) #Compute expectations of V'(a',s') using matrix multiplication
    
    c = (β.*EV_a).^(-1/γ) #consumption today
    a = (a′grid .+ c .- w̄.*ϵ')/(1+r̄)  #Implied assets today

    cf = Vector{Interpoland}(undef,Nϵ)#implied policy rules for each productivity
    for s in 1:Nϵ
        if a[1,s]> a̲
            c̲ = r̄*a̲ + w̄*ϵ[s]
            cf[s]= Interpoland(Basis(SplineParams([a̲; a[:,s]],0,1)),[c̲;c[:,s]])
        else
            cf[s]= Interpoland(Basis(SplineParams(a[:,s],0,1)),c[:,s])
        end
    end
    return cf
end

"""
    iteratebellman_time!(AM::AiyagriModel,Vcoefs)

Updates the coefficients of the value function using time iteration of the bellman equation
"""
function iteratebellman_time!(HH::HHModel)
    @unpack β,ϵ,Π,r̄,w̄,Nϵ,Vf,Φ = HH
    Vcoefs = vcat([Vf[s].coefs for s in 1:Nϵ]...)::Vector{Float64}
    agrid = nodes(Vf[1].basis)[1]
    Na = length(agrid)

    cf = computeoptimalconsumption(HH,Vcoefs) #Compute optimal consumption function
    c = zeros(Na*Nϵ) 
    EΦ = spzeros(Na*Nϵ,Na*Nϵ)
    for s in 1:Nϵ
        for s′ in 1:Nϵ
            c[(s-1)*Na+1:s*Na] = cf[s](agrid) #compute consumption at gridpoints
            a′ = (1+r̄)*agrid .+ ϵ[s]*w̄ .- c[(s-1)*Na+1:s*Na] #asset choice
            #Compute expectation of basis functions at a′
            EΦ[(s-1)*Na+1:s*Na,(s′-1)*Na+1:s′*Na] = Π[s,s′]*BasisMatrix(Vf[s].basis,Direct(),a′).vals[1][:]
        end
    end

    res = U(HH,c) .+ β.*EΦ*Vcoefs - Φ*Vcoefs
    Vcoefs′ = Φ\(U(HH,c) .+ β.*EΦ*Vcoefs)
    for s in 1:Nϵ
        Vf[s].coefs .= Vcoefs′[1+(s-1)*Na:s*Na]
    end
    return norm(res,Inf)
end

"""
iteratebellman_newton!(AM::AiyagriModel,Vcoefs)

Updates the coefficients of the value function using newton's method
"""
function iteratebellman_newton!(HH::HHModel)
    @unpack β,ϵ,Π,r̄,w̄,Nϵ,Vf,Φ = HH
    Vcoefs = vcat([Vf[s].coefs for s in 1:Nϵ]...)::Vector{Float64}
    agrid = nodes(Vf[1].basis)[1]
    Na = length(agrid)

    cf = computeoptimalconsumption(HH,Vcoefs) #Compute optimal consumption function
    c = zeros(Na*Nϵ) 
    EΦ = spzeros(Na*Nϵ,Na*Nϵ)
    for s in 1:Nϵ
        for s′ in 1:Nϵ
            c[(s-1)*Na+1:s*Na] = cf[s](agrid) #compute consumption at gridpoints
            a′ = (1+r̄)*agrid .+ ϵ[s]*w̄ .- c[(s-1)*Na+1:s*Na] #asset choice
            #Compute expectation of basis functions at a′
            EΦ[(s-1)*Na+1:s*Na,(s′-1)*Na+1:s′*Na] = Π[s,s′]*BasisMatrix(Vf[s].basis,Direct(),a′).vals[1][:]
        end
    end


    Jac = β.*EΦ .- Φ
    res = U(HH,c) .+ Jac*Vcoefs 
    Vcoefs′ = Vcoefs - Jac\res #newtons method
    for s in 1:Nϵ
        Vf[s].coefs .= Vcoefs′[1+(s-1)*Na:s*Na]
    end
    return norm(res,Inf)
end


"""
    solvebellman!(HH::HHModel)

Solves the bellman equation for given some initial 
value function V.
"""
function solvebellman!(HH::HHModel,tol=1e-8)
    #increases stability to iterate on the time dimension a few times
    diff = 1.
    for _ in 1:5
        iteratebellman_time!(HH)
    end
    while diff > tol
        #then use newtons method
        diff = iteratebellman_newton!(HH)
        println(diff)
    end
    Vcoefs = vcat([HH.Vf[s].coefs for s in 1:HH.Nϵ]...)::Vector{Float64}
    HH.cf = computeoptimalconsumption(HH,Vcoefs)
end

setupgrids_shocks!(HH,3.)
solvebellman!(HH);