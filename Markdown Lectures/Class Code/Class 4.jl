
function f(x)
    return x.^2
end

g(x) = x.^2

h = x -> x.^2

#plot(x->x.^2,0,1)



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

N = 50
A = 1.
α = 0.33
β = 0.95
kgrid = LinRange(0.01,0.5,N)
V,n_pol,k_pol = RBCbellmanmap(zeros(N),kgrid,A,α,β )

V - log.(A*kgrid.^α .- 0.01)