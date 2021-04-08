using Gadfly,QuantEcon
using Parameters



## Kalman Filter
@with_kw mutable struct KalmanFilter #The @with_kw allows us to given default values
    #Parameters
    A::Matrix{Float64} = [0.9][:,:]
    G::Matrix{Float64} = [1.][:,:]
    C::Matrix{Float64} = [1.][:,:]
    R::Matrix{Float64} = [1.][:,:]

    #Initial Beliefs
    x̂0::Vector{Float64} = [0.]
    Σ0::Matrix{Float64} = [1.][:,:]
end

kf = KalmanFilter()

kf = KalmanFilter(A=[0.9 0.1;0.7 0.6])


mc = rouwenhorst(5,0.9,1.)
P = mc.p
X̄ = collect(mc.state_values)


using LinearAlgebra
D,V = eigen(P')
println(D[5])
println(V[:,5])
P'*V[:,5] - V[:,5]

πstar = V[:,isapprox.(D,1)][:]
πstar ./= sum(πstar)#Need to normalize for probability
#Same as
πstar = πstar./sum(πstar)


P^1000


s_end = zeros(Int,10000)
for i in 1:10000
    s_end[i] = simulate_indices(mc,200,init=1)[end]
end



Θ = 2.
T=100
ϵ = randn(T+1)
y = [ϵ[t+1]-Θ*ϵ[t] for t in 1:T]

kf = KalmanFilter(A = [0. 0.;1. 0.],C=[1.;0.][:,:],G=[1 -Θ],R=zeros(1,1),x̂0=zeros(2),Σ0=Matrix{Float64}(I,2,2))

