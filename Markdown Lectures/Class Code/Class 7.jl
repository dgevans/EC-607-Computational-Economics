using QuantEcon


mc = rouwenhorst(5,0.95,0.01)
mc.state_values
mc.p


using QuantEcon,Parameters,DataFrames,Random,NLsolve

"""
An object containing all the model parameters for our DMP model,
with default values 0.99,0.212,0.72,0.4,0.05,0.72,0.83
"""
@with_kw mutable struct DMPModel
    β::Float64 = 0.99
    κ::Float64 = 0.212
    η::Float64 = 0.72
    b::Float64 = 0.4
    δ::Float64 = 0.05
    α::Float64 = 0.72
    A::Float64 = 0.83
    z::Vector{Float64}
    P::Matrix{Float64}
end

"""
DMPModel(Nz,ρ_z,σ_z) 

Constructs an instance of DMPModel from model parameters
"""
function DMPModel(Nz,ρ_z,σ_z)
    mc = rouwenhorst(Nz,ρ_z,σ_z)
    return DMPModel(z = exp.(mc.state_values), P = mc.p)
end

model = DMPModel(51,0.8,0.014)
println(model.β)
println(model.z)


p(dmp,θ) = dmp.A*θ.^(1-dmp.α) #note the .^
q(dmp,θ) = dmp.A*θ.^(-dmp.α)   

"""
    residuals(dmp::DMPModel,θvec)

Computes the residuals of the equilibrium conditions given θvec for a 
given set of parameters dmp
"""
function residuals(dmp::DMPModel,θvec)
    @unpack η,z,P,b,κ,δ,β = dmp
    S = length(z)
    qvec = q(dmp,θvec)
    #note @. means vectorize all operations
    rhs = @.((1-η)*(z -b) -η*κ*θvec+(1-δ)*κ/qvec)
    lhs = @.(κ/(β*qvec))
    return P*rhs -lhs
end

"""
    equilibrium(dmp::DMPModel)

Solves for the equilibrium of the DMP model with parameters in 
struct dmp
"""
function equilibrium(dmp::DMPModel)
    S = length(dmp.z)
    return nlsolve(x->residuals(dmp,x),ones(S)).zero
end

"""
    calibrate_κ!(dmp::DMPModel)

Internally calibrates κ to target an average θ of 1
"""
function calibrate_κ!(dmp::DMPModel)
    θvec0 = equilibrium(dmp) #first find the equilibrium for a given κ
    function res(κθvec)
        κ,θvec = κθvec[1],κθvec[2:end]
        dmp.κ = κ #note changing value form dmp
        return [1-(dmp.P^500*θvec)[1]; #long run θ is 1
                residuals(dmp,θvec)]#eqb residuals must be 0
    end
    κθvec0 = [dmp.κ;θvec] #

    dmp.κ =  nlsolve(res,κθvec0).zero[1]

    return equilibrium(dmp)
end

residuals(model,zeros(51))
θvec = equilibrium(model)
println(model.P^500*θvec)
θvec = calibrate_κ!(model)
println(model.P^500*θvec)

function simulate_economy(dmp,initial_state,T)
    s1,u1 = initial_state
    θvec = equilibrium(dmp)
    #preallocate space
    u,v,y,z,θ = zeros(T+1),zeros(T),zeros(T),zeros(T),zeros(T)
    u[1] = u1
    s = simulate_indices(MarkovChain(model.P),T,init=s1)
    for t in 1:T
        θ[t] = θvec[s[t]]
        z[t] = dmp.z[s[t]]
        y[t] = z[t]*(1-u[t])
        v[t] = θ[t]*u[t]
        u[t+1] = u[t]+dmp.δ*(1-u[t])-p(dmp,θ[t])*u[t]
    end
    #often convenient to return simulations as DataFrame
    return DataFrame(u=u[1:T],v=v,y=y,θ=θ,z=z)
end

df = simulate_economy(model,(25,0.057),200)