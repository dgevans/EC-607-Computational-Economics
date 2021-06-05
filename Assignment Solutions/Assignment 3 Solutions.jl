using BasisMatrices,LinearAlgebra,Parameters,Optim,Gadfly,FastGaussQuadrature,StatsBase,DataFrames

@with_kw mutable struct FirmProblem
    #Parameters
    γ::Float64 = 3.
    κ::Float64 = .05
    β::Float64 = 0.9981
    σ_ε::Float64 = 0.1
    N_ε::Int = 10 #number of quadrature points

    #Quadrature Nodes
    ε::Vector{Float64} = zeros(0)
    πε::Vector{Float64} = zeros(0)
    
    #Value function solutions
    V::Interpoland
    J::Float64
    μstar::Float64 = 0. #optimal markup
end

"""
    FirmProblem()   

Constructs the default calibration for the firm problem
"""
function FirmProblem(κ=.05,σ_ε=0.1,γ=3.,β=0.9981,Nμ=100)
    μstar = γ/(γ-1)
    μgrid = LinRange(μstar*0.2,μstar*2.,Nμ)
    V = Interpoland(SplineParams(μgrid,0,1),μ->(μ.-1).*μ.^(-γ)./(1-β))
    J = (μstar - 1)*μstar^(-γ)/(1-β) - κ
    
    fp = FirmProblem(γ=γ,κ=κ,β=β,V=V,J=J)
    ξ,w = gausshermite(fp.N_ε)
    fp.ε = sqrt(2)*σ_ε*ξ
    fp.πε = w./sum(w)
    fp.σ_ε = σ_ε
    return fp
end

function iterateBellman!(fp::FirmProblem,Vprime,Jprime)
    @unpack γ,β,ε,πε,κ = fp
    μbasis = Vprime.basis
    μgrid = nodes(μbasis)[1]
    Vgrid = zeros(length(μgrid))
    for (i,μ) in enumerate(μgrid)
        μ′ = μ./exp.(ε)
        Vgrid[i] = (μ-1)*μ^(-γ) + β*dot(πε,max.(Vprime(μ′),Jprime))
    end
    V = Interpoland(μbasis,Vgrid)

    res =  Optim.maximize(V,μgrid[1],μgrid[end])
    J = maximum(res) - κ
    fp.μstar = Optim.maximizer(res)
    return V,J
end

"""
    solveBellman!(fp::FirmProblem)

Solves the firm's bellman equation using fp.V and fp.J as initial values
"""
function solveBellman!(fp::FirmProblem)
    diff = 1.
    
    while diff > 1e-7
        V,J = iterateBellman!(fp,fp.V,fp.J)

        diff = norm(V.coefs-fp.V.coefs,Inf)

        fp.V = V
        fp.J = J
    end
end

"""
    simulateFirms(fp,N,μ0,T)

Simulate N firms for T periods until 
"""
function simulateFirms(fp,N,μ0,T)
    @unpack V,J,μstar,σ_ε,γ = fp

    μ = copy(μ0)
    df = DataFrame(zeros(T,3),[:P,:Δ,:frac_change])
    for t in 1:T
        μ ./= exp.(σ_ε.*randn(N)) #price shock
        change = J .> V(μ)
        
        Δ = @views μstar ./ μ[change] .- 1
        μ[change] .= μstar
        P = mean(μ.^(1-γ))^(1/(1-γ))
        df.P[t] = P
        df.Δ[t] = mean(abs.(Δ))
        df.frac_change[t] = sum(change)/N
    end

    #return path of aggregates and distribution
    return df,μ
end

#working calibration
fp = FirmProblem(0.003,0.045)
solveBellman!(fp)
N = 100_000
df,μ = simulateFirms(fp,N,1.5*ones(N),1000)
p1 = plot(df,y=:P)
p2 =plot(df,y=:Δ)
p3 = plot(df,y=:frac_change)
p4 = plot(x=μ,Geom.histogram)

df_irf,μ̂ = simulateFirms(fp,N,μ./1.05,30)