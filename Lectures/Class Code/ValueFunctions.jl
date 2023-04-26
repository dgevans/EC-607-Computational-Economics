## Analytical Example
using Plots,LinearAlgebra #plotting libraries
default(linewidth=2,legend=false,margin=5Plots.mm)
A = 1
α = 0.33
β = 0.95
J = 200 #number of iterations

E = zeros(J)
F = zeros(J)
E[1] = log(A)
F[1] = α
for j in 2:J
    #apply formula for E
    E[j] = β*E[j-1] + log(A/(1+β*F[j-1]))+
           β*F[j-1]*log(β*F[j-1]/(1+β*F[j-1])*A)
    #apply formula for F
    F[j] = α + α*β*F[j-1]
end
E_infty = (1-β)^(-1) * (log(A*(1-β*α))+
         (α*β)/(1-α*β)*log(A*β*α))
F_infty = α/(1-β*α)

#Now plot value functions
fig = plot( k->0 ,0.01,0.5,color=:lightblue,xlabel="Capital",ylabel="Value Function")
for j in 1:100
    #Add value function to plot for each iteration j in 1,2,..,100
    plot!(k->E[j]+F[j]*log(k),0.01,0.5,color=:lightblue)
end
#Add limiting value function
plot!(k->E_infty+F_infty*log(k),0.01,0.5,color=:red)

#Now plot the policy rules]
plot( k->0 ,0.01,0.5,color=:lightblue,xlabel="Capital",ylabel="Next Period Capital")
#NOTE only 10 iterations
for j in 1:10
    #Add policy rules to plot for each iteration j in 1,2,..,10
    plot!(k->(β*F[j])/(1+β*F[j])*A*k^α,0.01,0.5,color=:lightblue)
end
#Add limiting value function, note order = 1 puts it on top
plot!(k->(β*F_infty)/(1+β*F_infty)*A*k^α,0.01,0.5,color=:red)
#add line representing kprime = k
plot!(k->k,0.01,0.5,color=:black)


## McCall Search Model

"""
    mccallbellmanmap(v,w,π,c,β)

Iterates the McCall search model bellman equation for with value function v.
Returns the new value function.

# Arguments
* `v` vector of values for each wage
* `w` vector of wages
* `p` vector of probabilities for each wage
* `c` unemployment benefits
* `β` time discount factor

"""
function mccallbellmanmap(v,  w,p,c,β)
    #first compute value of rejecting the offer
    v_reject = c + β * dot(p,v) #note that this a Float (single number)
    #now compute value of accepting the wage offer
    v_accept = w/(1-β)
    
    #finally compare the two
    v_out = max.(v_reject,v_accept)
    #this is equivalent to
    S = length(w)
    for s in 1:S
        v_out[s] = max(v_reject,w[s]/(1-β))
    end
    
    return v_out
end

S = 40 #number of grid points
#uniform wage distribution between 1 10
w = LinRange(1,10,S)
p = ones(S)/S
β = 0.96 #lower β will mean code will converge faster
c = 3
v0 = zeros(S)

J = 50 #iterate code J times
V = zeros(J,S)
V[1,:] = mccallbellmanmap(v0,w,p,c,β)
for j in 2:J
    V[j,:] = mccallbellmanmap(V[j-1,:],w,p,c,β)
end

#Code to solve the McCall Model
"""
    solvemccall(w,π,c,β[,ϵ])

Iterates the McCall search model bellman equation until convergence criterion 
ϵ is reached

# Arguments
* `w` vector of wages
* `p` vector of probabilities for each wage
* `c` unemployment benefits
* `β` time discount factor
* `ϵ' Stopping criteria (default 1e-6)
"""
function solvemccall(w,p,c,β,ϵ=1e-6)
    #initialize
    v = w/(1-β)
    diff = 1.
    #check if stoping criteria is reached
    while diff > ϵ
        v_new = mccallbellmanmap(v,w,p,c,β)
        #use supremum norm
        diff = norm(v-v_new,Inf)
        v = v_new #reset v
    end
    return v
end 

#Test solution
v = solvemccall(w,p,c,β)
println(v - mccallbellmanmap(v,w,p,c,β))

scatter(w,v,xlabel="Wage",ylabel="Value Function")


# RBC Model
"""
   RBCbellmanmap(V,kgrid,A,α,β)

Iterates on the bellman equation for the standard neoclassical growth model

# Arguments
* `V` Vector of values for each capital level in kgrid
* `kgrid` Vector of capital levels
* `A` TFP
* `α` production function is A k^α
* `β` Time Discount factor
"""
function RBCbellmanmap(V,  kgrid,A,α,β)
    N = length(kgrid)
    V_out = zeros(N) #new value function
    n_pol = zeros(Int,N) #policy rule for grid points
    k_pol = zeros(N) #policy rule for capital
    obj = zeros(N) #objective to be maximized
    for n in 1:N #iterate for each initial capital
        for nprime in 1:N #iterate for choice of capital this period
            c = A*kgrid[n]^α - kgrid[nprime] #compute consumption
            if c <= 0
                obj[nprime] = -Inf #penalty if consumption <0
            else
                obj[nprime] = log(c)+β*V[nprime] #otherwise objective from RHS of bellman equation
            end
        end
        V_out[n],n_pol[n] = findmax(obj) #find optimal value and the choice that gives it
        k_pol[n] = kgrid[n_pol[n]] #record capital policy
    end
    return V_out,n_pol,k_pol
end




"""
    RBCsolve_bellman(V0,kgrid,A,α ,β[,ϵ=1e-6])

Solves the bellman equation by iterating until convergence

# Arguments
* `V0` Initial vector of values for each capital level in kgrid
* `kgrid` Vector of capital levels
* `A` TFP
* `α` production function is A k^α
* `β` Time Discount factor
* `ϵ` Convergence criteria
"""
function RBCsolve_bellman(V0,kgrid,A,α,β,ϵ=1e-6)
    diff = 1.
    V,n_pol,k_pol = RBCbellmanmap(V0,kgrid,A,α,β)
    while diff > ϵ
        V_new,n_pol,k_pol = RBCbellmanmap(V,kgrid,A,α,β)
        diff = norm(V_new-V,Inf)
        V = V_new 
    end
    return V,n_pol,k_pol
end

N = 500
A = 1.
α = 0.33
β = 0.95
kgrid = LinRange(0.01,0.5,N)
V,n_pol,k_pol = RBCsolve_bellman(zeros(N),kgrid,A,α,β )

#Plot value function
plot(k->E_infty+F_infty*log(k),.01,.5,color=:red,label="Analytical",legend=true)
scatter!(kgrid,V,label="Approximation",color=:lightblue,xlabel="Capital",ylabel="Value Function")


#plot capital policy rules
plot(k->(β*F_infty)/(1+β*F_infty)*A*k^α,.01,.5,color=:red,label="Analytical",legend=true)
scatter!(kgrid,k_pol,label="Approximation",color=:lightblue,xlabel="Capital",ylabel="Capital Next Period")



## Extensions
N = 50
kgrid = LinRange(0.01,0.5,N)
@time RBCsolve_bellman(zeros(N),kgrid,A,α,β);

N = 500
kgrid = LinRange(0.01,0.5,N)
@time RBCsolve_bellman(zeros(N),kgrid,A,α,β);


##Simulating Capital
"""
    simulate_k(n_0,T,n_pol,kgrid)

Simulates the path of capital given policy rule n_pol and 
initial capital state kgrid[n_0] for T periods
"""
function simulate_k(n_0,T,n_pol,kgrid)
    k = zeros(T+1) # capital stock
    n = zeros(Int,T+1) #index of the capital stock
    n[1] = n_0
    k[1] = kgrid[n_0]
    for t in 1:T
        n[t+1] = n_pol[n[t]] #get the policy rule for the index
        k[t+1] = kgrid[n[t+1]]
    end

    return k
end
plot(simulate_k(1,25,n_pol,kgrid),xlabel="Time",ylabel="Capital Stock")

# Howard improvement algorithm
"""
    RBCbellmanmap_howard(V,nprime,kgrid,A,α,β)

Iterates on the bellman equation for the standard neoclassical growth model using policies nprime,
rather than computing the optimal policies

# Arguments
* `V` Vector of values for each capital level in kgrid
* `n_pol` policy rules k[n_pol[n]] is the capital choice when previous period capital is k[n] 
* `kgrid` Vector of capital levels
* `A` TFP
* `α` production function is A k^α
* `β` Time Discount factor
"""
function RBCbellmanmap_howard(V,n_pol,kgrid,A,α,β)
    N = length(kgrid)
    V_new = zeros(N)
    for n in 1:N
        #use given policy 
        c = A*kgrid[n]^α - kgrid[n_pol[n]]
        V_new[n] = log(c) + β*V[n_pol[n]]
    end
    return V_new
end;


"""
    RBCsolve_bellman_howard(V0,H,kgrid,A,α ,β[,ϵ=1e-6])

Solves the bellman equation by iterating until convergence. Uses howard improvement algorithm:
only solves for optimal policy every H iteration

# Arguments
* `V0` Initial vector of values for each capital level in kgrid
* `H` Controls how frequently optimal policy is solved, H=1 implies every period
* `kgrid` Vector of capital levels
* `A` TFP
* `α` production function is A k^α
* `β` Time Discount factor
* `ϵ` Convergence criteria
"""
function RBCsolve_bellman_howard(V0,H,kgrid,A,α,β;ϵ=1e-6)
    diff = 1.
    V,n_pol,k_pol = RBCbellmanmap(V0,kgrid,A,α,β)
    #do 5 or so iterations first to allow policys to converge
    for j in 1:5
        V_new,n_pol,k_pol = RBCbellmanmap(V,kgrid,A,α,β)
        V = V_new
    end
    #Now apply the Howard Improvement Algorithm
    while diff > ϵ
        V_old = V 
        for h in 1:H
            V_new = RBCbellmanmap_howard(V,n_pol,kgrid,A,α,β)
            V = V_new
        end
        #perform one iteration updating policies
        V_new,n_pol,k_pol = RBCbellmanmap(V,kgrid,A,α,β)
        diff = norm(V_new-V_old,Inf)
        V = V_new
    end
    return V,n_pol,k_pol
end;


#Test
N = 50
kgrid = LinRange(0.01,0.5,N)
#evaluate once to compile
RBCsolve_bellman_howard(zeros(N),100,kgrid,A,α,β);
#Test Timing
@time RBCsolve_bellman_howard(zeros(N),100,kgrid,A,α,β);
@time RBCsolve_bellman(zeros(N),kgrid,A,α,β);

N = 500
kgrid = LinRange(0.01,0.5,N)
#evaluate once to compile
RBCsolve_bellman_howard(zeros(N),100,kgrid,A,α,β);
#Test Timing
@time RBCsolve_bellman_howard(zeros(N),100,kgrid,A,α,β);
@time RBCsolve_bellman(zeros(N),kgrid,A,α,β);


#Uncertainty
A = [0.97,1.03]
#Transition matrix
Π = [0.6 0.4;0.4 0.6]

"""
    RBCbellmanmap_stochastic(V,kgrid,A,Π,α,β,U)

Iterates on the bellman equation for the standard neoclassical growth model

# Arguments
* `V` Vector of values for each capital level in kgrid
* `kgrid` Vector of capital levels
* `A` Vector of TFP values for each state
* `Π` Transition matrix
* `α` production function is A k^α
* `β` Time Discount factor
"""
function RBCbellmanmap_stochastic(V,kgrid,A,Π,α,β)
    N = length(kgrid) #Number of gridpoints of capital
    S = length(A) #Number of stochastic states
    V_new = zeros(S,N) #New Value function
    n_pol = zeros(Int,S,N) #New policy rule for grid points
    k_pol = zeros(S,N) #New policy rule for capital
    obj = zeros(N) #objective to be maximized
    EV = Π*V #precompute expected value for speed
    for n in 1:N
        for s in 1:S
            for nprime in 1:N
                c = A[s]*kgrid[n]^α - kgrid[nprime] #compute consumption
                if c <= 0
                    obj[nprime] = -Inf #punish if c <=0
                else
                    obj[nprime] = log(c) + β*EV[s,nprime] #otherwise compute objective
                end
            end
            #find optimal value and policy
            V_new[s,n],n_pol[s,n] = findmax(obj)
            k_pol[s,n] = kgrid[n_pol[s,n]]
        end
    end
    return V_new,n_pol,k_pol
end;


"""
    RBCsolve_bellman_stochastic(V0,kgrid,A,α ,β[,ϵ=1e-6])

Solves the bellman equation by iterating until convergence

# Arguments
* `V0` Initial vector of values for each capital level in kgrid
* `kgrid` Vector of capital levels
* `A` Vector of TFP values for each state
* `Π` Transition matrix
* `α` production function is A k^α
* `β` Time Discount factor
* `ϵ` Convergence criteria
"""
function RBCsolve_bellman_stochastic(V0,kgrid,A,Π,α,β,ϵ=1e-6)
    diff = 1.
    V,n_pol,k_pol = RBCbellmanmap_stochastic(V0,kgrid,A,Π,α,β)
    while diff > ϵ
        V_new,n_pol,k_pol = RBCbellmanmap_stochastic(V,kgrid,A,Π,α,β)
        diff = norm(V_new-V,Inf)
        V = V_new
    end
    return V,n_pol,k_pol
end;

N = 500
kgrid = LinRange(0.01,0.5,N)
V,n_pol,k_pol = RBCsolve_bellman_stochastic(zeros(2,N),kgrid,A,Π,α,β);

plot(legend=true)
for s in 1:2
    plot!(k->α*β*A[s]*k^α,.01,.5,color=:red,label="Analytical")
    scatter!(kgrid,k_pol[s,:],color=:lightblue,label="Approximation")
end
xlabel!("Capital")
ylabel!("Capital Next Period")


##simulation
using QuantEcon
"""
    simulate_k_stochastic(n_0,T,n_pol,kgrid,Π)

Simulates the path of capital given policy rule n_pol and 
initial capital state kgrid[n_0] and aggregate state s_0 for T periods
"""
function simulate_k_stochastic(n_0,s0,T,n_pol,kgrid,Π)
    k = zeros(T+1) # capital stock
    n = zeros(Int,T+1) #index of the capital stock
    s = simulate_indices(MarkovChain(Π),T;init=s0)
    n[1] = n_0
    k[1] = kgrid[n_0]
    for t in 1:T
        n[t+1] = n_pol[s[t],n[t]] #get the policy rule for the index
        k[t+1] = kgrid[n[t+1]]
    end

    return k
end

scatter(simulate_k_stochastic(150,1,50,n_pol,kgrid,Π),xlabel="Time",ylabel="Capital Stock")


"""
    simulate_k_stochastic(n_0,T,n_pol,kgrid,Π)

Simulates the path of capital given policy rule n_pol and 
initial capital state kgrid[n_0] and aggregate state s_0 for T periods
"""
function simulate_k_stochastic_withc(n_0,s0,T,n_pol,kgrid,Π,A,α)
    k = zeros(T+1) # capital stock
    y = zeros(T)
    c = zeros(T)
    n = zeros(Int,T+1) #index of the capital stock
    s = simulate_indices(MarkovChain(Π),T;init=s0)
    n[1] = n_0
    k[1] = kgrid[n_0]
    for t in 1:T
        n[t+1] = n_pol[s[t],n[t]] #get the policy rule for the index
        k[t+1] = kgrid[n[t+1]]
        y[t] = A[s[t]]*k[t]^α
        c[t] = y[t] - k[t+1] 
    end

    return k,c,y
end



kpath,cpath,ypath =simulate_k_stochastic_withc(150,1,50,n_pol,kgrid,Π,A,α)

T=50
plot(1:T,kpath[1:T],layout=(3,1))
plot!(1:T,cpath,subplot=2)
plot!(1:T,ypath,subplot=3)