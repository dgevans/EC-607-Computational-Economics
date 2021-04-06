meantosses = [mean(flipNcoins(N,0.5)) for N in 1:100];

#equivalent to
meantosses = zeros(100)
for N in 1:100
    meantosses[N] = mean(flipNcoins(N,0.5))
end

using Distributions
numheads = [sum(flipNcoins(15,0.5)) for k in 1:100_000]

plot(x=numheads,Geom.histogram(density=true,bincount=15))
true_density = pdf.(Binomial(15,0.5),0:15)
plot(x=0:15,y=pdf.(Binomial(15,0.5),0:15),color=[colorant"red"],Geom.point)

plot(layer(x=numheads,Geom.histogram(density=true,bincount=15)),
     layer(x=0:15,y=pdf.(Binomial(15,0.5),0:15),color=[colorant"red"],Geom.line,order=1)
    ,Guide.XLabel("# of Heads"),Guide.YLabel("Probability"))



μ = 0.
ρ = 0.9
σ = 1.
T = 10
x0 = 0.

x = zeros(T+1)# initialize
x[1] = x0
for t in 1:T
    x[t+1] = (1-ρ)*μ + ρ*x[t] + σ*randn()
end


x[2:5]
#compare to
[x[2],x[3],x[4],x[5]]

A = rand(3,3)

@views A[2:3,1] #produces view(A,2:3,1)


#Create types

mutable struct AR1
    μ::Float64 #Mean of the AR(1)
    ρ::Float64 #persistence of the AR(1)
    σ::Float64 #standard deviaiton of the AR(1)
end



using Parameters
@unpack μ,σ = ar1
#the same as
μ = ar.μ
σ = ar.σ

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

T = 100
N = 1000
X = zeros(T,N)
for i in 1:N
    X[:,i] .= simulateAR1(ar1,0.,T)
end

plot(X,x=Row.index,y=Col.value,color=Col.index,Geom.line,Guide.XLabel("Time"))