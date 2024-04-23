using Plots,LinearAlgebra
default(linewidth=2,legend=false,margin=10Plots.mm)
A = 1
α = 0.33
β = 0.95
J = 200 #number of iterations

kgrid = collect(range(0.01,0.5,100))
V = zeros(length(kgrid)) #initialize value function

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

