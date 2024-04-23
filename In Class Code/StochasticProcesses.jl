rand()

using Random

Random.seed!(12345) #can put any integer here

matrix = rand(2, 2)


flips = zeros(100) #Where to store the flips
r = rand(100) #A vector of 100 uniform random numbers on [0,1]
for i in 1:100
    flips[i] = r[i] < 0.5 #check to see if the i th flip is heads
end
println(flips)


A = rand(2,2)
b = rand(2)


using Distributions

"""
    flipNcoins(N,p=0.5)

Flips N coins with probability p of returning 1 (heads)
"""
function flipNcoins(N,p=0.5)
    return rand(N) .< p
end

println(flipNcoins(15))

println(flipNcoins(15,1.))

using Distributions
dist  = LogNormal(0,1)

println(mean(dist))
println(std(dist))
println(quantile(dist,0.5))
println(cdf(dist,1.))
println(pdf(dist,1.))
println(rand(dist))

using Statistics
mean(flipNcoins(50))

mean(flipNcoins(50,0.3))

using Plots

default(linewidth=2,legend=false,margin=5Plots.mm)


Nmax = 1000
meantosses = [mean(flipNcoins(N,0.5)) for N in 1:Nmax]; #This is called a list compression

scatter(1:Nmax,meantosses,xlabel="Tosses",ylabel="Average # Heads")

toss_range = 5:1000
meantosses = [mean(flipNcoins(N,0.5)) for N in toss_range]; #This is called a list compression
stdtosses = [std(flipNcoins(N,0.5)) for N in toss_range]; #This is called a list compression
plot(toss_range,meantosses,layout = (2,1),subplot=1,ylabel="Average # Heads")
plot!(toss_range,stdtosses,layout = (2,1),subplot=2,xlabel="Tosses",ylabel="STD # Heads")



"""
   simulateAR1(ρ,σ,T)

Simulates an AR(1) with mean μ, persistence ρ, and standard deviation σ for 
T periods with initial value x0
"""
function simulateAR1(μ,ρ,σ,x0,T)
    x = zeros(T+1)# initialize
    x[1] = x0
    for t in 1:T
        x[t+1] = (1-ρ)*μ + ρ*x[t] + σ*randn()
    end
    return x[2:end]
end

using Random
Random.seed!(12345) #can put any integer here
plot(1:100,simulateAR1(0,0.,1,0,100),xlabel="Time",ylabel="AR(1)")


mutable struct AR1
    μ::Float64 #Mean of the AR(1)
    ρ::Float64 #persistence of the AR(1)
    σ::Float64 #standard deviaiton of the AR(1)
end

ar1 = AR1(0.,0.8,1.) #Note order matters here
println(ar1.ρ)

using Parameters



"""
   simulateAR1(ar,x0,T)

Simulates an AR(1) ar for T periods with initial value x0
"""
function simulateAR1(ar,x0,T)
    @unpack σ,μ,ρ = ar #note order doesn't matter
    x = zeros(T+1)# initialize
    x[1] = x0
    for t in 1:T
        x[t+1] = (1-ρ)*μ + ρ*x[t] + σ*randn()
    end
    return x[2:end]
end
plot(1:100,simulateAR1(ar1,0.,100),xlabel="Time",ylabel="AR(1)")

T = 50
N = 1000
X = zeros(T,N)
for i in 1:N
    X[:,i] .= simulateAR1(ar1,2.,T)
end


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
