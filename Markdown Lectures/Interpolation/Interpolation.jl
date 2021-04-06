using Gadfly, LinearAlgebra
#make lines and points thicker and bigger labels
Gadfly.push_theme(Theme(major_label_font_size=20pt,minor_label_font_size=14pt,key_label_font_size=16pt,
                        line_width=2pt,point_size=3pt))



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


#Let's see how they look
xvec = LinRange(0,1,9)
layers = [layer(x->B1(x,k,xvec),0,1) for k in 1:9]
plot(layers...,Guide.xlabel("x"),Guide.ylabel("Linear Basis Function"))


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

plot(layer(x->f(x),0,1,color=["True Function"]),
    layer(x->fhat(x,f,xvec),0,1,color=["Approximation"]),
    Guide.xlabel("x"),Guide.ylabel("f(x)"))



## Higher Order Splines
using BasisMatrices
#Most basic usage
xvec = LinRange(0,1,4) #break points for the B splines
fhat1 = Interpoland(SplineParams(xvec,0,1),f) #linear Interpolation
fhat3 = Interpoland(SplineParams(xvec,0,3),f) #cubic Interpolation

plot(layer(x->f(x),0,1,color=["True Function"]),
    layer(x->fhat1(x),0,1,color=["Linear Approx."]),
    layer(x->fhat3(x),0,1,color=["Cubic Approx."]),
    Guide.xlabel("x"),Guide.ylabel("f(x)"))


#What happens with kinks
f2(x) = x < 0.5 ? 0 : x - 0.5 # returns 0 if x < 0.5 otherwise x - 0.5
fhat1 = Interpoland(SplineParams(xvec,0,1),x->f2.(x))
fhat3 = Interpoland(SplineParams(xvec,0,3),x->f2.(x))
plot(layer(x->f2.(x),0,1,color=["True Function"]),
    layer(x->fhat1(x),0,1,color=["Linear Approx."]),
    layer(x->fhat3(x),0,1,color=["Cubic Approx."]),
    Guide.xlabel("x"),Guide.ylabel("f(x)"))


## Monomial Basis
layers = [layer(x->x.^k,0,1,color=["x^$(k-1)"]) for k in 1:9]
plot(layers...,Guide.xlabel("x"),Guide.ylabel("x^k"))


#Let's try fitting the Runge function
n = 9
Phi = zeros(n,n)
xvec = LinRange(-1,1,n)
f_runge(x) = 1 ./(1 .+ 25 .* x.^2)
yvec = zeros(n)
for i in 1:n
    for k in 1:n
        Phi[i,k] = xvec[i]^(k-1)
    end
    yvec[i] = f_runge(xvec[i])
end

w = Phi\yvec #computes coefficients for Interpolation
det(Phi) ##note nearly sigular

#How good is the fit?
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
plot(layer(f_runge,-1,1,color=["Runge"]),
     layer(x->fhatmonomial(x,w),-1,1,color=["Monomial"]),
     Guide.xlabel("x"),Guide.ylabel("f(x)"))



## Chebyshev Polynomials
T(x,k) = cos.((k-1).*acos.(x))
layers = [layer(x->T(x,k),-1,1,color=["Order $(k-1)"]) for k in 1:9]
plot(layers...,Guide.xlabel("x"),Guide.ylabel("Chebyshev Polynomials")) 


##Using Basis matrix
chebbasis = Basis(ChebParams(9,-1,1))
xnodes = nodes(chebbasis)[1] #Gives nodes for the chebyshev Polynomials (zeros)
plot(layer(x=xnodes,y=xnodes.*0,color=["Nodes"]),
     layer(x->T(x,10),-1,1,color=["Chebyshev Polynomial"]),
     Guide.xlabel("x"),Guide.colorkey(title="")) 



f_cheb = Interpoland(chebbasis,f_runge)
plot(layer(f_runge,-1,1,color=["Runge"]),
     layer(x->fhatmonomial(x,w),-1,1,color=["monomial"]),
     layer(x->f_cheb(x),-1,1,color=["Chebyshev"]),
     Guide.xlabel("x"),Guide.ylabel("f(x)"),Guide.colorkey(title=""))

plot(layer(x->fhatmonomial(x,w)-f_runge(x),-1,1,color=["Monomial"]),
    layer(x->f_cheb(x)-f_runge(x),-1,1,color=["Chebyshev"]),
    Guide.xlabel("x"),Guide.ylabel("Errors"),Guide.colorkey(title=""))

#Compare to Splines
xgrid = LinRange(-1,1,9) #use 9 grid points like chebyshev
f_spline = Interpoland(SplineParams(xgrid,0,3),f_runge)

plot(layer(f_runge,-1,1,color=["Runge"]),
     layer(x->f_spline(x),-1,1,color=["Cubic Spline"]),
     layer(x->f_cheb(x),-1,1,color=["Chebyshev"]),
     Guide.xlabel("x"),Guide.ylabel("f(x)"),Guide.colorkey(title=""))

plot(layer(x->f_spline(x)-f_runge(x),-1,1,color=["Cubic Spline"]),
     layer(x->f_cheb(x)-f_runge(x),-1,1,color=["Chebyshev"]),
     Guide.xlabel("x"),Guide.ylabel("Errors"),Guide.colorkey(title=""))



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
    V = map(k->optimalpolicy(para,Vprime,k).V,knodes)
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
kprime,V = solvebellman(para,getV0(para))

#Check accuracy
@unpack A,α,β = para
kmin,kmax = para.kgrid[1],para.kgrid[end]
plot(layer(k->α.*β.*A.*k.^α,kmin,kmax,color=["Truth"]),
     layer(k->kprime(k),kmin,kmax,color=["Approximation"]),
     Guide.xlabel("Capital"), Guide.ylabel("Future Capital"),Guide.colorkey(title=""))

    
plot(k->kprime(k).-α.*β.*A.*k.^α,kmin,kmax,Guide.xlabel("Capital"), Guide.ylabel("Future Capital Error"))

#Compare different order's of splines
para.spline_order = 1
kprime1,_ = solvebellman(para,getV0(para))
para.spline_order = 2
kprime2,_ = solvebellman(para,getV0(para))
para.spline_order = 3
kprime3,_ = solvebellman(para,getV0(para))

plot(layer(k->α.*β.*A.*k.^α,kmin,kmax,color=["Truth"]),
     layer(k->kprime1(k),kmin,kmax,color=["Linear Spline"]),
     layer(k->kprime2(k),kmin,kmax,color=["Quadratic Spline"]),
     layer(k->kprime3(k),kmin,kmax,color=["Cubic Spline"]),
     Guide.xlabel("Capital"), Guide.ylabel("Future Capital"),Guide.colorkey(title=""))

#Check Errors
plot(layer(k->kprime1(k).-α.*β.*A.*k.^α,kmin,kmax,color=["Linear Spline"]),
     layer(k->kprime2(k).-α.*β.*A.*k.^α,kmin,kmax,color=["Quadratic Spline"]),
     layer(k->kprime3(k).-α.*β.*A.*k.^α,kmin,kmax,color=["Cubic Spline"]),
     Guide.xlabel("Capital"), Guide.ylabel("Future Capital"),Guide.colorkey(title=""))
