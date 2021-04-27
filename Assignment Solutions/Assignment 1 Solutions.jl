
using Parameters,Distributions,LinearAlgebra,Gadfly,Random
@with_kw struct MAKalmanFilter #Not using mutable because I don't ever want to change the values
    #Parameters
    θ::Float64 #Will need to set this parameter to construct the Kalman Fiter
    A::Matrix{Float64} = [0. 0.;
                          1. 0.]
    G::Matrix{Float64} = [1. -θ]
    C::Matrix{Float64} = [1.;0][:,:]
    R::Matrix{Float64} = [0.][:,:]

    #Initial Beliefs
    x̂0::Vector{Float64} = [0.;0]
    Σ0::Matrix{Float64} = Matrix{Float64}(I,2,2) #This constructs the identity matrix
end

"""
    updateBeliefs(KF::KalmanFilter,y,x̂,Σ)

Uses the Kalman Filter to update beliefs x̂,Σ using data y 
"""
function updateBeliefs(KF::MAKalmanFilter,y,x̂,Σ)
    @unpack A,G,C,R = KF
    a = y - G*x̂
    K = A*Σ*G'*inv(G*Σ*G'+R)
    x̂′= A*x̂ + K*a
    Σ′ = C*C' + K*R*K' + (A-K*G)*Σ*(A' - G'*K')
    
    return x̂′,Σ′
end

"""
    applyFilter(KF::KalmanFilter,y)

Applies the Kalman Filter on data y. Assume y is mxT where 
T is the number of periods
"""
function applyFilter(KF::MAKalmanFilter,y)
    @unpack x̂0,Σ0 = KF

    T = size(y,2) #how many rows are y
    x̂ = zeros(length(x̂0),T+1)
    Σ = zeros(length(x̂0),length(x̂0),T+1)
    x̂[:,1] .= x̂0
    Σ[:,:,1] .= Σ0
    for t in 1:T
        x̂[:,t+1],Σ[:,:,t+1] = updateBeliefs(KF,y[:,t],x̂[:,t],Σ[:,:,t])
    end

    return x̂,Σ
end

"""
    simulateMA1(θ,T)

Simulates the MA1 process with lag coefficient θ for T periods
"""
function simulateMA1(θ,T)
    ϵ = randn(T+1) #need to draw t+1 values
    
    return (y=[ϵ[t+1]-θ*ϵ[t] for t in 1:T],ϵ=ϵ) #note ϵ[t] is really ϵ_{t-1}
end;


"""
    applyFilterMA(y,θ)

Applys the filter to data y given parameter θ
"""
function applyFilterMA(y::Vector{Float64},θ::Float64)
    kf = MAKalmanFilter(θ=θ)#construct KalmanFilter object with parameter θ
    return applyFilter(kf,y')
end

"""
    get_loglikelihood(y,θ)

Constructs the log likelihood for a given parameter θ
"""
function get_loglikelihood(y,θ)
    T = length(y)
    kf = MAKalmanFilter(θ=θ)
    x̂,Σ = applyFilterMA(y,θ)
    logl = 0.
    for t in 1:T
        Gx̂ = (kf.G*x̂[:,t])[1] #make it a float for normal distribution
        Ω = (kf.G*Σ[:,:,t]*kf.G' + kf.R)[1]
        logf = log(pdf(Normal(Gx̂,sqrt(Ω)),y[t]))
        logl += logf
    end

    return logl
end


y,ϵ = simulateMA1(2,100)

x̂,Σ = applyFilterMA(y,2.)
plot(layer(y=ϵ[1:end-1],Geom.line,color=["Truth"]),
    layer(y=x̂[2,:],Geom.line,color=["Filter"]),Guide.ylabel("Time"))

plot(θ->get_loglikelihood(y,θ),-4.,4)
plot(θ->get_loglikelihood(y,θ),1.5,3)
y = simulateMA1(2,10000)

plot(θ->get_loglikelihood(y,θ),1.5,3.)

θrange = LinRange(1.2,3.,100)
Random.seed!(45634)
y = simulateMA1(2,100)
logl_100 = [get_loglikelihood(y,θ) for θ in θrange]
logl_100max = maximum(logl_100)
Random.seed!(45634)
y = simulateMA1(2,1000)
logl_1000 = [get_loglikelihood(y,θ) for θ in θrange]
logl_1000max = maximum(logl_1000)


plot(layer(x=θrange,y=logl_100./abs(logl_100max),Geom.line,color=["T=100"]),
     layer(x=θrange,y=logl_1000./abs(logl_1000max),Geom.line,color=["T=1000"]),
     Guide.XLabel("θ"),Guide.YLabel("Log Likelihood Relative to Max"))