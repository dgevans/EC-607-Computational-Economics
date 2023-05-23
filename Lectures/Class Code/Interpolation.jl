using Plots
default(linewidth=2,legend=false)

"""
    B1(x,k,xvec)

Constructs the kth linear B-splines using knot points xvec.
"""
function B1(x,k,xvec)
    n = length(xvec)
    #check if first condition is satisfied
    if x <= xvec[k] && k>1 && x >= xvec[k-1]
        return (x-xvec[k-1])/(xvec[k]-xvec[k-1])
    end
    #check if second condition
    if x >= xvec[k] && k < n && x <= xvec[k+1]
        return (xvec[k+1]-x)/(xvec[k+1]-xvec[k])
    end
    #otherwise return 0.
    return 0.
end

xvec = LinRange(0,1,9)
plt = plot(xlabel="x",ylabel="Linear Basis Function")
for k in 1:9
    plot!(x->B1(x,k,xvec),0,1,color=:lightblue)
end
plt

#Using linear B-Splines
f(x) = log.(x.+0.1) # our function to interpolate
xvec = LinRange(0,1,5) #knots for the B splines
function fhat(x,f,xvec)
    K = length(xvec)
    ret = 0.
    for k in 1:K
        ret += f(xvec[k])*B1(x,k,xvec)
    end
    return ret
end
plot(x->f(x),0,1,label="True Function",legend=true)
plot!(x->fhat(x,f,xvec),0,1,label="Approximation")

xvec2 = [0.,0.1,0.2,0.5,1.]
plot!(x->fhat(x,f,xvec2),0,1,label="Approximation2")

plt = plot(xlabel="x",ylabel="Linear Basis Function")
for k in 1:5
    plot!(x->B1(x,k,xvec2),0,1,color=:lightblue)
end
plt

using BasisMatrices
#Most basic usage
xvec = LinRange(0,1,6) #break points for the B splines
fhat1 = Interpoland(SplineParams(xvec,0,1),f) #linear Interpolation
xvec = LinRange(0,1,5)
fhat2 = Interpoland(SplineParams(xvec,0,2),f) #cubic Interpolation

plot(x->f(x),0,1,label="True Function",legend=true)
plot!(x->fhat1(x),0,1,label="Linear Approx.")
plot!(x->fhat2(x),0,1,label="Quadratic Approx.",ylabel="f(x)",xlabel="x")

xvec = LinRange(0,1,10)
qbasis = SplineParams(xvec,0,2)
Φ = BasisMatrix(Basis(qbasis),Direct(),LinRange(0,1,1000)).vals[1]'
plt = scatter(nodes(qbasis),0*nodes(qbasis))
for i in 1:size(Φ,1)
    plot!(LinRange(0,1,1000),Φ[i,:])
end
plt

plot(x->f(x),0,2,label="True Function",legend=true)
plot!(x->fhat1(x),0,2,label="Linear Approx.")
plot!(x->fhat3(x),0,2,label="Cubic Approx.",ylabel="f(x)",xlabel="x")

f2(x) = x < 0.5 ? 0 : x - 0.5 # returns 0 if x < 0.5 otherwise x - 0.5
xvec = [0.,0.3,0.4,0.45,0.5,0.55,0.6,0.8,1.]
fhat1 = Interpoland(SplineParams(xvec,0,1),x->f2.(x))
fhat3 = Interpoland(SplineParams(xvec,0,3),x->f2.(x))
plot(x->f2(x),0,1,label="True Function",legend=true)
plot!(x->fhat1(x),0,1,label="Linear Approx.")
plot!(x->fhat3(x),0,1,label="Cubic Approx.",ylabel="f(x)",xlabel="x")

plt = plot(ylabel="x^k",xlabel="x",legend=true)
for k in 1:9
    plot!(x->x.^k,0,1,label="x^$(k-1)")
end
plt

using LinearAlgebra
n = 9
Phi = zeros(n,n)
xvec = LinRange(-1,1,n)
f_runge(x) = 1 ./(1 .+ 25 .* x.^2) #The Runge Function
yvec = zeros(n)
for i in 1:n
    for k in 1:n
        Phi[i,k] = xvec[i]^(k-1)
    end
    yvec[i] = f_runge(xvec[i])
end
w = Phi\yvec #computes coefficients for Interpolation
det(Phi) #note nearly sigular

"""
    fhatmonomial(x,w::Array{Float64})
Computes ∑_k w_k x^(k-1)
"""
function fhatmonomial(x,w::Array{Float64})
    n = length(w)
    fhat = 0.
    for k in 1:n
        fhat += w[k]*x^(k-1)
    end
    return fhat
end
plot(f_runge,-1,1,label="Runge",legend=true)
plot!(x->fhatmonomial(x,w),-1,1,label="Monomial",xlabel="x",ylabel="f(x)")

T(x,k) = cos.((k-1).*acos.(x))
plt = plot(xlabel="x",ylabel="Chebyshev Polynomials",legend=true)
for k in 1:9
    plot!(x->T(x,k),-1,1,label="Order $(k-1)")
end
plt

#using BasisMatrix code
chebbasis = Basis(ChebParams(20,-1,1))
xnodes = nodes(chebbasis)[1] #Gives nodes for the chebyshev Polynomials (zeros)
plot(x->T(x,10),-1,1,label="Chebyshev Polynomial")
scatter!(xnodes,0 .* xnodes,label="Nodes",legend=true)

f_cheb = Interpoland(chebbasis,f_runge)
plot(f_runge,-1,1,label="Runge",legend=true)
plot!(x->fhatmonomial(x,w),-1,1,label="monomial")
plot!(x->f_cheb(x),-1,1,label="Chebyshev",xlabel="x",ylabel="f(x)")

plot(x->fhatmonomial(x,w)-f_runge(x),-1,1,label="Monomial",legend=true)
plot!(x->f_cheb(x)-f_runge(x),-1,1,label="Chebyshev",xlabel="x",ylabel="Errors")

xgrid = LinRange(-1,1,9) #use 9 grid points like chebyshev
f_spline = Interpoland(SplineParams(xgrid,0,3),f_runge)
plot(f_runge,-1,1,label="Runge",legend=true)
plot!(x->f_spline(x),-1,1,label="Cubic Spline")
plot!(x->f_cheb(x),-1,1,label="Chebyshev",xlabel="x",ylabel="f(x)")

plot(x->f_spline(x)-f_runge(x),-1,1,label="Cubic Spline",legend=true)
plot!(x->f_cheb(x)-f_runge(x),-1,1,label="Chebyshev",xlabel="x",ylabel="Errors")

using Parameters
@with_kw mutable struct NCParameters
    A::Float64 = 1.
    α::Float64 = 0.3
    β::Float64 = 0.96
    kgrid::Vector{Float64} = LinRange(0.05,0.5,20)
    spline_order::Int = 3
end

using Optim
"""
    optimalpolicy(para::NCParameters,Vprime,k)

Computes optimal policy using continuation value function V and current capital
level k given parameters in para.
"""
function optimalpolicy(para::NCParameters,Vprime,k)
    @unpack A,α,β,kgrid = para
    k_bounds = [kgrid[1],kgrid[end]]
    f_objective(kprime) = -( log(A*k^α-kprime)+β*Vprime(kprime) ) #stores objective as function
    k_max = min(A*k^α-.001,k_bounds[2]) #Can't have negative consumptions
    result = optimize(f_objective,k_bounds[1],k_max)
    return (kprime = result.minimizer,V=-result.minimum) #using named tuples 
end;

"""
    bellmanmap(Vprime,para::NCParameters)

Apply the bellman map given continuation value function Vprime
"""
function bellmanmap(para::NCParameters,Vprime::Interpoland)
    kbasis = Vprime.basis
    #sometimes it's helpful to tell julia what type a variable is
    knodes = nodes(kbasis)[1]::Vector{Float64}
    TV = k->optimalpolicy(para,Vprime,k).V
    V = TV.(knodes)
    return Interpoland(kbasis,V)
end;


function bellmanmap_naive(para::NCParameters,Vprime)
    return k->optimalpolicy(para,Vprime,k).V
end

"""
    solvebellman(para::NCParameters,V0::Interpoland)

Solves the bellman equation for a given V0
"""
function solvebellman(para::NCParameters,V0::Interpoland)
    diff = 1
    #Iterate of BellmanMap until difference in coefficients goes to zero
    n = 0
    while diff > 1e-6
        V = bellmanmap(para,V0)
        diff = norm(V.coefs-V0.coefs,Inf)
        V0 = V
        n +=1
        println(n)
    end
    kbasis = V0.basis
    knodes = nodes(kbasis)[1]::Vector{Float64}
    #remember optimalpolicy also returns the argmax
    k′ = k->optimalpolicy(para,V0,k).kprime
    #Now get policies
    return Interpoland(kbasis,k′.(knodes)),V0
end;

"""
    getV0(para::NCParameters)

Initializes V0(k) = 0 using the kgrid of para
"""
function getV0(para::NCParameters)
    @unpack kgrid,spline_order = para

    kbasis = Basis(SplineParams(kgrid,0,spline_order))

    return Interpoland(kbasis,k->0 .*k)
end
para = NCParameters()
para.kgrid = LinRange(0.05,0.5,20)
kprime,V = solvebellman(para,getV0(para));

ktruth(para,k) = para.α.*para.β.*para.A.*k.^para.α
kmin,kmax = para.kgrid[1],para.kgrid[end]
plot(k->ktruth(para,k),kmin,kmax,label="Truth",xlabel="Capital",ylabel="Future Capital")
plot!(k->kprime(k),kmin,kmax,label="Approximation",legend=true)

plot(k->kprime(k).-ktruth(para,k),kmin,kmax,xlabel="Capital", ylabel="Future Capital Error")

para.spline_order = 1
kprime1,_ = solvebellman(para,getV0(para))
para.spline_order = 2
kprime2,_ = solvebellman(para,getV0(para))
para.spline_order = 3
kprime3,_ = solvebellman(para,getV0(para))

plot(k->ktruth(para,k),kmin,kmax,label="Truth",legend=true)
plot!(k->kprime1(k),kmin,kmax,label="Linear Spline")
plot!(k->kprime2(k),kmin,kmax,label="Quadratic Spline")
plot!(k->kprime3(k),kmin,kmax,label="Cubic Spline",xlabel="Capital",ylabel="Future Capital")

plot(k->kprime1(k).-ktruth(para,k),kmin,kmax,label="Linear Spline",legend=true)
plot!(k->kprime2(k).-ktruth(para,k),kmin,kmax,label="Quadratic Spline")
plot!(k->kprime3(k).-ktruth(para,k),kmin,kmax,label="Cubic Spline",xlabel="Capital",ylabel="Future Capital")


basis_x = SplineParams(LinRange(-1,1,5),0,3) #cubic splines along x
basis_y = ChebParams(3,-1,1)#Chebyshev polynomials along y

basis = Basis(basis_x,basis_y)

X = nodes(basis)[1]

f2d = x-> exp(-x[1]^2-x[2]^2)
#compute f at each node
fvals = [f2d(X[i,:]) for i in 1:size(X,1)]

f̂ = Interpoland(basis,fvals);

plot(x->f2d([x,0]),-1,1,label="Function",legend=true)
plot!(x->f̂([x,0]),-1,1,label="Approximation",xlabel="x",ylabel="f(x)")

plot(y->f2d([0,y]),-1,1,label="Function",legend=true)
plot!(y->f̂([0,y]),-1,1,label="Approximation",xlabel="y",ylabel="f(x)")

plot(x->f2d([x,0.5]),-1,1,label="Function",legend=true)
plot!(x->f̂([x,0.5]),-1,1,label="Approximation",xlabel="x",ylabel="f(x)")

using Roots
using Distributions
σ = 2. #Standard
γ = 2. #Targets Frisch elasticity of 0.5
σ_α = sqrt(0.147) #Taken from HSV
N = 1000
alphaDist = Normal(-σ_α^2/2,σ_α)
αvec = rand(alphaDist,N);

"""
    approximate_household_labor(NlŴ,NT,σ,γ)

Approximates HH policies as a function of log after tax wage and transfers.
"""
function approximate_household_labor(NlŴ,NT,σ_α,σ,γ)
    lŴbasis = ChebParams(NlŴ,-5*σ_α+log(1-.8),5*σ_α)
    Tbasis = ChebParams(NT,0.,2.) #we know optimal tax will always be positive
    basis = Basis(lŴbasis,Tbasis)
    X = nodes(basis)[1]
    N = size(X,1) #How many nodes are there?
    c,h = zeros(N),zeros(N)
    for i in 1:N 
        Ŵ,T = exp(X[i,1]),X[i,2]
        res(h) = (Ŵ*h+T)^(-σ)*Ŵ-h^γ
        min_h = max(0,(.0000001-T)/Ŵ) #ensures c>.0001
        h[i] = fzero(res,min_h,20000.) #find hours that solve HH problem
        c[i] = Ŵ*h[i]+T
    end
    U = @. c^(1-σ)/(1-σ)-h^(1+γ)/(1+γ)
    return Interpoland(basis,c),Interpoland(basis,h),Interpoland(basis,U)
end;

"""
    budget_residual(τ,T,hf)

Computes the residual of the HH budget constraint given policy (τ,T)
"""
function budget_residual(τ,T,αvec,hf)
    N = length(αvec)
    X = [αvec .+ log(1-τ)  T*ones(N)]
    tax_income = sum(hf(X).*exp.(αvec).*τ)/N
    return tax_income - T
end;

"""
    government_welfare(τ,T,αvec,σ,γ)

Solves for government welfare given tax rate τ
"""
function government_welfare(τ,T,αvec,Uf)
    X = [αvec .+ log(1-τ)  T*ones(N)]
    return sum(Uf(X))/N
end;

using NLopt
""" 
    find_optimal_policy(αvec,Uf,hf)

Computes the optimal policy given policy fuctions hf and indirect utility Uf
"""
function find_optimal_policy(αvec,Uf,hf)
    opt = Opt(:LN_COBYLA, 2)
    lower_bounds!(opt, [0., 0.])
    upper_bounds!(opt, [0.5,Inf])
    ftol_rel!(opt,1e-8)

    min_objective!(opt, (x,g)->-government_welfare(x[1],x[2],αvec,Uf))
    equality_constraint!(opt, (x,g) -> -budget_residual(x[1],x[2],αvec,hf))

    minf,minx,ret = NLopt.optimize(opt, [0.3, 0.3])
    if ret == :FTOL_REACHED
        return minx
    end
end;

cf,hf,Uf = approximate_household_labor(10,10,σ_α,σ,γ)
find_optimal_policy(αvec,Uf,hf)
@time find_optimal_policy(αvec,Uf,hf)
#Remember the old code took 2 seconds

cf,hf,Uf = approximate_household_labor(5,5,σ_α,σ,γ)
find_optimal_policy(αvec,Uf,hf)

cf,hf,Uf = approximate_household_labor(10,10,σ_α,σ,γ)
find_optimal_policy(αvec,Uf,hf)

cf,hf,Uf = approximate_household_labor(20,20,σ_α,σ,γ)
find_optimal_policy(αvec,Uf,hf)