
using BasisMatrices,Gadfly
#Most basic usage
f(x) = log.(x.+0.1) # our function to interpolate
xvec = LinRange(0,1,10) #break points for the B splines
fhat1 = Interpoland(SplineParams(xvec,0,1),f) #linear Interpolation
fhat3 = Interpoland(SplineParams(xvec,0,3),f) #cubic Interpolation

plot(layer(x->f(x),0,1,color=["True Function"]),
    layer(x->fhat1(x),0,1,color=["Linear Approx."]),
    layer(x->fhat3(x),0,1,color=["Cubic Approx."]),
    Guide.xlabel("x"),Guide.ylabel("f(x)"),Guide.colorkey(title=""))

plot(layer(x->f(x),0,10.,color=["True Function"]),
    layer(x->fhat1(x),0.,10.,color=["Linear Approx."]),
    layer(x->fhat3(x),0,10.,color=["Cubic Approx."]),
    Guide.xlabel("x"),Guide.ylabel("f(x)"),Guide.colorkey(title=""))


qbasis = SplineParams(xvec,0,3)

Φ = BasisMatrix(Basis(qbasis),Direct(),LinRange(0,1,1000)).vals[1]'
plot(layer(x=nodes(qbasis),y=0*nodes(qbasis)),
    [layer(x=LinRange(0,1,1000),y=Φ[i,:],Geom.line,color=["$i"]) for i in 1:size(Φ,1)]...,
     xintercept=xvec,Geom.vline(color="white"))
     #)

f2(x) = x < 0.5 ? 0 : x - 0.5 # returns 0 if x < 0.5 otherwise x - 0.5
fhat1 = Interpoland(SplineParams(xvec,0,1),x->f2.(x))
fhat3 = Interpoland(SplineParams(xvec,0,3),x->f2.(x))
plot(layer(x->f2.(x),0,1,color=["True Function"]),
    layer(x->fhat1(x),0,1,color=["Linear Approx."]),
    layer(x->fhat3(x),0,1,color=["Cubic Approx."]),
    Guide.xlabel("x"),Guide.ylabel("f(x)"),Guide.colorkey(title=""))





## Value Function Interation
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
end

"""
    bellmanmap(Vprime,para::NCParameters)

Apply the bellman map given continuation value function Vprime
"""
function bellmanmap(para::NCParameters,Vprime::Interpoland)
    kbasis = Vprime.basis
    #sometimes it's helpful to tell julia what type a variable is
    knodes = nodes(kbasis)[1]::Vector{Float64}
    TV = k->optimalpolicy(para,Vprime,k).V
    V = map(TV,knodes)
    return Interpoland(kbasis,V)
end

"""
    solvebellman(para::NCParameters,V0::Interpoland)

Solves the bellman equation for a given V0
"""
function solvebellman(para::NCParameters,V0::Interpoland)
    diff = 1
    #Iterate of Bellman Map until difference in coefficients goes to zero
    while diff > 1e-6
        V = bellmanmap(para,V0)
        diff = norm(V.coefs-V0.coefs,Inf)
        V0 = V
    end
    kbasis = V0.basis
    knodes = nodes(kbasis)[1]::Vector{Float64}
    #remember optimalpolicy also returns the argmax
    kprime = map(k->optimalpolicy(para,V0,k).kprime,knodes)
    #Now get policies
    return Interpoland(kbasis,kprime),V0
end

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
V0 = getV0(para)



#Compare different order's of splines
@unpack A,α,β = para
kmin,kmax = para.kgrid[1],para.kgrid[end]
para.spline_order = 1
kprime1,_ = solvebellman(para,getV0(para))
para.spline_order = 2
kprime2,_ = solvebellman(para,getV0(para))
para.spline_order = 3
kprime3,_ = solvebellman(para,getV0(para))

chebbasis = ChebParams(15,kmin,kmax)
V0 = Interpoland(chebbasis,k->0 .* k)
kprime_cheb,_ = solvebellman(para,V0)

plot(layer(k->α.*β.*A.*k.^α,kmin,kmax,color=["Truth"]),
     layer(k->kprime1(k),kmin,kmax,color=["Linear Spline"]),
     layer(k->kprime2(k),kmin,kmax,color=["Quadratic Spline"]),
     layer(k->kprime3(k),kmin,kmax,color=["Cubic Spline"]),
     layer(k->kprime_cheb(k),kmin,kmax,color=["Chebyshev"]),
     Guide.xlabel("Capital"), Guide.ylabel("Future Capital"),Guide.colorkey(title=""))


plot(layer(k->kprime2(k).-α.*β.*A.*k.^α,kmin,kmax,color=["Quadratic Spline"]),
     layer(k->kprime3(k).-α.*β.*A.*k.^α,kmin,kmax,color=["Cubic Spline"]),
     layer(k->kprime_cheb(k).-α.*β.*A.*k.^α,kmin,kmax,color=["Chebyshev"]),
     Guide.xlabel("Capital"), Guide.ylabel("Future Capital"),Guide.colorkey(title=""))



basis_x = SplineParams(LinRange(-1,1,5),0,3) #cubic splines along x
basis_y = SplineParams(LinRange(-1,1,5),0,3) #cubic splines along x

basis = Basis(basis_x,basis_y)

xynodes = nodes(basis)
X = xynodes[1]
#note
X[:,1]
#same as
kron(ones(3),xynodes[2][1])


X = nodes(basis)[1]
f2d = x-> exp(-x[1]^2-x[2]^2)
fvals = [f2d(X[i,:]) for i in 1:size(X,1)]

f̂ = Interpoland(basis,fvals)

#at a node
plot(layer(x->f2d([x,0]),-1,1,color=["Function"]),
     layer(x->f̂([x,0]),-1,1,color=["Approximation"]),
     Guide.xlabel("x"),Guide.ylabel("f"))

#away from a node
plot(layer(x->f2d([x,0.5]),-1,1,color=["Function"]),
     layer(x->f̂([x,0.5]),-1,1,color=["Approximation"]),
     Guide.xlabel("x"),Guide.ylabel("f"))



# Recall
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
end



"""
    budget_residual(τ,T,hf)

Computes the residual of the HH budget constraint given policy (τ,T)
"""
function budget_residual(τ,T,αvec,hf)
    N = length(αvec)
    X = [αvec .+ log(1-τ)  T*ones(N)]
    tax_income = sum(hf(X).*exp.(αvec).*τ)/N
    return tax_income - T
end


"""
    government_welfare(τ,T,αvec,σ,γ)

Solves for government welfare given tax rate τ
"""
function government_welfare(τ,T,αvec,Uf)
    X = [αvec .+ log(1-τ)  T*ones(N)]
    return sum(Uf(X))/N
end

cf,hf,Uf = approximate_household_labor(10,10,σ_α,σ,γ)

τ = 0.3
T = .4

N = length(αvec)
X = [αvec .+ log(1-τ)  T*ones(N)]

hf(X)

budget_residual(τ,T,αvec,hf)
government_welfare(τ,T,αvec,Uf)




