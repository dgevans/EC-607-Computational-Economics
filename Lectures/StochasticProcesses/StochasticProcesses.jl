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


#Distributions package
using Distributions
dist  = LogNormal(0,1)
println(mean(dist))
println(std(dist))
println(quantile(dist,0.5))
println(cdf(dist,1.))
println(pdf(dist,1.))
println(rand(dist))

using Plots
default(linewidth=2,legend=false)

meantosses = [mean(flipNcoins(N,0.5)) for N in 1:100]; #This is called a list compression
plot(1:100,meantosses,xlabel="Tosses",ylabel="Average # Heads")


toss_range = 5:1000
meantosses = [mean(flipNcoins(N,0.5)) for N in toss_range]; #This is called a list compression
stdtosses = [std(flipNcoins(N,0.5)) for N in toss_range]; #This is called a list compression
plot(toss_range,meantosses,layout = (2,1),subplot=1,ylabel="Average # Heads")
plot!(toss_range,stdtosses,layout = (2,1),subplot=2,xlabel="Tosses",ylabel="STD # Heads")


numheads = [sum(flipNcoins(15,0.5)) for k in 1:100_000]
histogram(numheads,bins=25,normalize=:probability,xlabel="# of Heads",ylabel="Probability")
plot!(0:15,pdf.(Binomial(15,0.5),0:15))


## Continuous StochasticProcesses
"""
   simulateAR1(??,??,T)

Simulates an AR(1) with mean ??, persistence ??, and standard deviation ?? for 
T periods with initial value x0
"""
function simulateAR1(??,??,??,x0,T)
    x = zeros(T+1)# initialize
    x[1] = x0
    for t in 1:T
        x[t+1] = (1-??)*?? + ??*x[t] + ??*randn()
    end
    return x[2:end]
end

plot(1:100,simulateAR1(0.,0.8,1.,0,100),xlabel="Time",ylabel="AR(1)")

mutable struct AR1
    ??::Float64 #Mean of the AR(1)
    ??::Float64 #persistence of the AR(1)
    ??::Float64 #standard deviaiton of the AR(1)
end

ar1 = AR1(0.,0.8,1.) #Note order matters here
println(ar1.??)


"""
   simulateAR1(ar,x0,T)

Simulates an AR(1) ar for T periods with initial value x0
"""
function simulateAR1(ar,x0,T)
    x = zeros(T+1)# initialize
    x[1] = x0
    for t in 1:T
        x[t+1] = (1-ar.??)*ar.?? + ar.??*x[t] + ar.??*randn()
    end
    return x[2:end]
end
plot(1:100,simulateAR1(ar1,0.,100),xlabel="Time",ylabel="AR(1)")


using Parameters
"""
   simulateAR1(ar,x0,T)

Simulates an AR(1) ar for T periods with initial value x0
"""
function simulateAR1(ar,x0,T)
    @unpack ??,??,?? = ar #note order doesn't matter
    x = zeros(T+1)# initialize
    x[1] = x0
    for t in 1:T
        x[t+1] = (1-??)*?? + ??*x[t] + ??*randn()
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
plot(1:T,X,xlabel="Time")

plot(1:T,mean(X,dims=2),ylabel="Mean",layout=(2,1),subplot=1)
plot!(1:T,std(X,dims=2),xlabel="Time",ylabel="Standard Deviation",subplot=2)


## Kalman Filter
@with_kw mutable struct KalmanFilter #The @with_kw allows us to given default values
    #Parameters
    A::Matrix{Float64} = [0.9][:,:]
    G::Matrix{Float64} = [1.][:,:]
    C::Matrix{Float64} = [1.][:,:]
    R::Matrix{Float64} = [1.][:,:]

    #Initial Beliefs
    x??0::Vector{Float64} = [0.]
    ??0::Matrix{Float64} = [1.][:,:]
end

"""
    updateBeliefs(KF::KalmanFilter,y,x??,??)

Uses the Kalman Filter to update beliefs x??,?? using data y 
"""
function updateBeliefs(KF::KalmanFilter,y,x??,??)
    @unpack A,G,C,R = KF
    a = y - G*x??
    K = A*??*G'*inv(G*??*G'+R)
    x?????= A*x?? + K*a
    ????? = C*C' + K*R*K' + (A-K*G)*??*(A' - G'*K')
    
    return x?????,?????
end

"""
    applyFilter(KF::KalmanFilter,y)

Applies the Kalman Filter on data y. Assume y is mxT where 
T is the number of periods
"""
function applyFilter(KF::KalmanFilter,y)
    @unpack x??0,??0 = KF

    T = size(y,2) #how many rows are y
    x?? = zeros(length(x??0),T+1)
    ?? = zeros(length(x??0),length(x??0),T+1)
    x??[:,1] .= x??0
    ??[:,:,1] .= ??0
    for t in 1:T
        x??[:,t+1],??[:,:,t+1] = updateBeliefs(KF,y[:,t],x??[:,t],??[:,:,t])
    end

    return x??,??
end

#applying Kalman Filter
KF = KalmanFilter(A=[0.95][:,:],C=[1.][:,:], G=[2.][:,:],R=[1.][:,:],x??0=[0.],??0=[1/(1-0.95)][:,:])
#First generate data
T = 99
x = simulateAR1(0.0,0.95,1.,1.,T)
y = 2*x .+ randn(T)

#Now perform Kalman filter
x??,?? = applyFilter(KF,y') #note we need y to be a row

plot(layer(y=x,Geom.line,color=["True State"]),layer(y=x??,Geom.line,color=["Filter"]),
        Guide.YLabel("x"),Guide.XLabel("Time"),Guide.colorkey(title=""))

plot(1:T,x,xlabel="Time",ylabel="x",label="True State",legend=true)
plot!(1:T,x??[1:T],label="Filter")


#What happens if we increase G
x = simulateAR1(0.0,0.95,1.,1.,T)
y = 20*x .+ randn(T)
KF.G .= 20.
#Now perform Kalman filter
x??,?? = applyFilter(KF,y') #note we need y to be a row

plot(1:T,x,xlabel="Time",ylabel="x",label="True State",legend=true)
plot!(1:T,x??[1:T],label="Filter")

#What happens if we decrease G
x = simulateAR1(0.0,0.95,1.,1.,T)
y = 0.2*x .+ randn(T)
KF.G .= 0.2
#Now perform Kalman filter
x??,?? = applyFilter(KF,y') #note we need y to be a row

plot(1:T,x,xlabel="Time",ylabel="x",label="True State",legend=true)
plot!(1:T,x??[1:T],label="Filter")



## Finite state Markov chains
using QuantEcon,LinearAlgebra
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

P,X?? = mc_ar1.p,mc_ar1.state_values
println((P^14*X??)[1])


D,V = eigen(P')  #should be left unit eigenvector
??star = V[:,isapprox.(D,1)][:]
??star ./= sum(??star)#Need to normalize

#or 

??star2 = (P^200)[15,:] #probability distribution 1000 periods in the future
println(norm(??star -??star2))


@time eigen(P');
@time ??star2 = (P^200)[15,:]


s_end = zeros(Int,10000)
for i in 1:10000
    s_end[i] = simulate_indices(mc_ar1,200,init=1)[end]
end

histogram(s_end,bins=51,normalize=:probability,xlabel="State",ylabel="Probability")
plot!(1:51,??star2)
