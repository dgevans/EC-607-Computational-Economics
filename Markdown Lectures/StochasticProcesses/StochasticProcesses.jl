## Simple random numbers
rand()
rand([10,20])
rand(2,2)

flip = rand() < 0.5

flips = zeros(100) #Where to store the flips
r = rand(100) #A vector of 100 uniform random numbers on [0,1]
for i in 1:100
    flips[i] = r[i] < 0.5 #check to see if the i th flip is heads
end
println(flips)

flips = []
for i in 1:100
    push!(flips,rand() < 0.5) #check to see if the i th flip is heads
end
println(flips)

flips = zeros(3) #Where to store the flips
flips_alt = zeros(3)
r = rand(3) #A vector of 100 uniform random numbers on [0,1]

for i in 1:3
    flips[i] = r[i] < 0.5 #check to see if the i th flip is heads
end
i = 1 
flips_alt[i] = r[i] < 0.5
i = 2 
flips_alt[i] = r[i] < 0.5
i = 3 
flips_alt[i] = r[i] < 0.5
println(flips)
println(flips_alt)


flips = rand(100) .< 0.5 #note the . won't work without it
println(flips)


"""
    flipNcoins(N,p=0.5)

Flips N coins with probability p of returning 1 (heads)
"""
function flipNcoins(N,p=0.5)
    return rand(N) .< p
end

println(flipNcoins(15))

println(flipNcoins(15,1.))


## Some Monte Carlo
using Statistics
mean(flipNcoins(50))
mean(flipNcoins(50,0.3))


using Gadfly
set_default_plot_size(12inch, 6inch) #set size
#make lines and points thicker and bigger labels
Gadfly.push_theme(Theme(major_label_font_size=20pt,minor_label_font_size=14pt,key_label_font_size=16pt,
                        line_width=2pt,point_size=3pt))

meantosses = [mean(flipNcoins(N,0.5)) for N in 1:100]; #This is called a list compression
plot(x=1:100,y=meantosses,Guide.XLabel("Tosses"),Guide.YLabel("Average # Heads"))

meantosses = [mean(flipNcoins(N,0.5)) for N in 5:1000]; #This is called a list compression
stdtosses = [std(flipNcoins(N,0.5)) for N in 5:1000]; #This is called a list compression
hstack(plot(x=5:1000,y=meantosses,Geom.line,Guide.XLabel("Tosses"),Guide.YLabel("Average # Heads")),
       plot(x=5:1000,y=stdtosses,Geom.line,Guide.XLabel("Tosses"),Guide.YLabel("STD # Heads"))) 


       numheads = [sum(flipNcoins(15,0.5)) for k in 1:100_000]
       plot(x=numheads,Geom.histogram(density=true,numbins=25),Guide.XLabel("# of Heads"),Guide.YLabel("Probability"))



## Continuous StochasticProcesses
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

plot(x=1:100,y=simulateAR1(0.,0.8,1.,0,99),Geom.line,Guide.XLabel("Time"),Guide.YLabel("AR(1)"))

mutable struct AR1
    μ::Float64 #Mean of the AR(1)
    ρ::Float64 #persistence of the AR(1)
    σ::Float64 #standard deviaiton of the AR(1)
end

ar1 = AR1(0.,0.8,1.) #Note order matters here
println(ar1.ρ)


"""
   simulateAR1(ar,x0,T)

Simulates an AR(1) ar for T periods with initial value x0
"""
function simulateAR1(ar,x0,T)
    x = zeros(T+1)# initialize
    x[1] = x0
    for t in 1:T
        x[t+1] = (1-ar.ρ)*ar.μ + ar.ρ*x[t] + ar.σ*randn()
    end
    return x[2:end]
end
plot(x=1:100,y=simulateAR1(ar1,0.,99),Geom.line,Guide.XLabel("Time"),Guide.YLabel("AR(1)"))


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
plot(x=1:100,y=simulateAR1(ar1,0.,99),Geom.line,Guide.XLabel("Time"),Guide.YLabel("AR(1)"))



T = 50
N = 1000
X = zeros(T,N)
for i in 1:N
    X[:,i] .= simulateAR1(ar1,0.,T)
end
mean(X,2)


## Kalman Filter
@with_kw mutable struct KalmanFilter #The @with_kw allows us to given default values
    #Parameters
    A::Matrix{Float64} = [0.9][:,:]
    G::Matrix{Float64} = [1.][:,:]
    C::Matrix{Float64} = [1.][:,:]
    R::Matrix{Float64} = [1.][:,:]

    #Initial Beliefs
    x̂0::Vector{Float64} = [0.][:,:]
    Σ0::Matrix{Float64} = [1.]
end

"""
    updateBeliefs(KF::KalmanFilter,y,x̂,Σ)

Uses the Kalman Filter to update beliefs x̂,Σ using data y 
"""
function updateBeliefs(KF::KalmanFilter,y,x̂,Σ)
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
function applyFilter(KF::KalmanFilter,y)
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

#applying Kalman Filter
KF = KalmanFilter(A=[0.95][:,:],C=[1.][:,:], G=[2.][:,:],R=[1.][:,:],x̂0=[0.],Σ0=[1/(1-0.95)][:,:])
#First generate data
T = 99
x = simulateAR1(0.0,0.95,1.,1.,T)
y = 2*x .+ randn(T)

#Now perform Kalman filter
x̂,Σ = applyFilter(KF,y') #note we need y to be a row

plot(layer(y=x,Geom.line,color=["True State"]),layer(y=x̂,Geom.line,color=["Filter"]),
        Guide.YLabel("x"),Guide.XLabel("Time"),Guide.colorkey(title=""))


#What happens if we increase G
x = simulateAR1(0.0,0.95,1.,1.,T)
y = 20*x .+ randn(T)
KF.G .= 20.
#Now perform Kalman filter
x̂,Σ = applyFilter(KF,y') #note we need y to be a row

plot(layer(y=x,Geom.line,color=["True State"]),layer(y=x̂,Geom.line,color=["Filter"]),
        Guide.YLabel("x"),Guide.XLabel("Time"),Guide.colorkey(title=""))
    

#What happens if we decrease G
x = simulateAR1(0.0,0.95,1.,1.,T)
y = 0.2*x .+ randn(T)
KF.G .= 0.2
#Now perform Kalman filter
x̂,Σ = applyFilter(KF,y') #note we need y to be a row

plot(layer(y=x,Geom.line,color=["True State"]),layer(y=x̂,Geom.line,color=["Filter"]),
        Guide.YLabel("x"),Guide.XLabel("Time"),Guide.colorkey(title=""))



## Finite state Markov chains
using QuantEcon
P = [0.6 0.4;
     0.4 0.6]
s = simulate(MarkovChain(P),100,init=1)
println(s)

mc_ar1 = rouwenhorst(51,0.9,0.014)

X = zeros(15,1000)
for i in 1:1000
    X[:,i] = simulate(mc_ar1,15,init=1)
end
println(mean(X[15,:]))

P,X̄ = mc_ar1.p,mc_ar1.state_values
println((P^14*X̄)[1])


D,V = eigen(P')  #should be left unit eigenvector
πstar = V[:,isapprox.(D,1)][:]
πstar ./= sum(πstar)#Need to normalize

#or 

πstar2 = (P^200)[15,:] #probability distribution 1000 periods in the future
println(norm(πstar -πstar2))


@time eigen(P');
@time πstar2 = (P^1000)[15,:]


s_end = zeros(Int,10000)
for i in 1:10000
    s_end[i] = simulate_indices(mc_ar1,200,init=1)[end]
end

plot(layer(x=s_end,Geom.histogram(density=true,bincount=51)),
     layer(x=1:51,y=πstar2,color=[colorant"red"],Geom.point,order=1)
    ,Guide.XLabel("State"),Guide.YLabel("Probability"))