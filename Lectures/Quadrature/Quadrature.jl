using Plots
default(linewidth=2,legend=false)

using BasisMatrices
f_runge(x) = 1 ./(1 .+ 25 .* x.^2) #The Runge Function
N = 10
xgrid = collect(LinRange(-1,1,10))
fhat = Interpoland(Basis(SplineParams(xgrid,0,1)),f_runge)
plot(f_runge,-1,1,label="Runge Function",legend=false)
plot!(x->fhat(x),-1,1,label="Approximation",xlabel="x",ylabel="f(x)")

"""
    linear_quadrature(f,a,b,N)

Computes integral using quadrature based on linear interpolation
"""
function linear_quadrature(f,a,b,N)
    xgrid = LinRange(a,b,N)

    #Ask me: Why am I doing it like this?
    return (b-a)*(0.5*f(xgrid[1])+sum(f(x) for x in xgrid[2:end-1])+0.5*f(xgrid[end]))/N
end

plot(logN->linear_quadrature(f_runge,-1,1,Int(round(10^logN))),1,6,
    xlabel="Log Base 10",ylabel="Quadrature")

using FastGaussQuadrature
ξ,w = gausslegendre(5)
println(ξ)
println(w);

"""
    gausslegendre_quadrature(f,a,b,N)

Computes integral using Gauss Legendre quadrature
"""
function gausslegendre_quadrature(f,a,b,N)
    ξ,w = gausslegendre(N)
    #Next rescale nodes to be on [a,b]
    x = ξ.*(b-a)/2 .+ b/2 .+ a/2

    #return weighted sum (note need to scale based on (b-a))
    return sum(w[i]*f(x[i]) for i in 1:N)*(b-a)/2
end;

plot(logN->linear_quadrature(f_runge,-1,1,Int(round(10^logN))),1,6,label="Linear Rule",legend=true)
plot!(logN->gausslegendre_quadrature(f_runge,-1,1,Int(round(10^logN))),1,6,label="Gauss Legendre",
    xlabel="Log Base 10",ylabel="Quadrature")

"""
    expectation_normal(h,μ,σ,N)

Compute the expectation of a function h(y) of a normal variable y∼N(μ,σ) using
N point Gauss-Hermite quadrature
"""
function expectation_normal(h,μ,σ,N)
    ξ,w = gausshermite(N)
    x = sqrt(2).*σ.*ξ .+ μ

    return sum(w[i]*h(x[i]) for i in 1:N)/sqrt(pi)
end;

scatter(1:10,expectation_normal.(exp,0.,1.,1:10),xlabel="# nodes")

ξx,wx = gausslegendre(10)
ξy,wy = gausslegendre(20)

ξ = hcat(kron(ones(length(ξy)),ξx),kron(ξy,ones(length(ξx))))
w = kron(wy,wx)

scatter(ξ[:,1],ξ[:,2])

"""
    gausslegendre_quadrature(f,a,b,N)

Computes integral using Gauss Legendre quadrature
"""
function gausslegendre_quadrature(f,a::Vector,b::Vector,N::Vector)
    D = length(N)
    #dimension 1
    ξ,W = gausslegendre(N[1])
    X = ξ.*(b[1]-a[1])/2 .+ b[1]/2 .+ a[1]/2 #X is going to be an N^D×D vector

    for i in 2:D
        ξ,w = gausslegendre(N[i])
        #Next rescale nodes to be on [a,b]
        x = ξ.*(b[i]-a[i])/2 .+ b[i]/2 .+ a[i]/2

        X = hcat(kron(ones(N[i]),X),kron(x,ones(size(X,1))))
        W = kron(w,W)
    end
    #return weighted sum (note need to scale based on (b-a))
    return sum(W[i]*f(X[i,:]) for i in 1:size(X,1))*prod((b.-a)./2)
end;

using LinearAlgebra
f_runge_ND(x) = 1 /(1 + 25 * norm(x)) #The Runge Function

plot(logN->gausslegendre_quadrature(f_runge_ND,-1*ones(2),1*ones(2),Int(round(10^logN))*ones(Int,2)),1,3,
    xlabel="#Nodes Log Base 10",ylabel="Quadrature")

maxD = 7
time = zeros(maxD)
for D in 1:maxD
  time[D] = @elapsed gausslegendre_quadrature(f_runge_ND,-1*ones(D),1*ones(D),10*ones(Int,D)) 
end
plot(1:maxD,time,xlabel="Dimensions",ylabel="Time (s)")

x = rand(100000)*2 .- 1 #uniform on [-1,1]
Ef = sum(f_runge.(x))/length(x)
println(Ef*2) #don't forget to scale by (b-a)

gausslegendre_quadrature(f_runge,-1,1,1000)

"""
    montecarlo_integration(f,a,b,N)

Computes integral using Monte Carlo methods
"""
function montecarlo_integration(f,a::Vector,b::Vector,N)
    D = length(a)
    X = rand(D,N).*(b.-a) .+ a
    #return weighted sum (note need to scale based on (b-a))
    return sum(f(X[:,i]) for i in 1:N)/N*prod(b.-a)
end;

using DataFrames,StatsPlots
Nmax = 10^6
dfMonte = DataFrame()
dfMonte.N = 1000:10000:Nmax
for D in 1:10
    dfMonte[!,"dimension$D"] = [montecarlo_integration(f_runge_ND,-1*ones(D),ones(D),N) for N in 1000:10000:Nmax]
    dfMonte[!,"dimension$D"] ./= dfMonte[end,"dimension$D"] #normalize by last point
end
@df dfMonte plot(:N,cols(2:11),legend=true)

using QuasiMonteCarlo
ξx,wx = gausslegendre(10)
ξy,wy = gausslegendre(10)
XQuad = hcat(kron(ones(length(ξy)),ξx),kron(ξy,ones(length(ξx))))'
XUnif = rand(2,100).*2 .- 1
XSobol = QuasiMonteCarlo.sample(100,[-1,-1],[1,1],SobolSample())
scatter(XQuad[1,:],XQuad[2,:],label="10x10 Legendre",legend=true)
scatter!(XUnif[1,:],XUnif[2,:],label="Uniform Sampler")
scatter!(XSobol[1,:],XSobol[2,:],label="Sobol Sampler")

"""
    quasi_montecarlo_integration(f,a,b,N)

Computes integral using Quasi Monte Carlo Methods
"""
function quasi_montecarlo_integration(f,a::Vector,b::Vector,N)
    X = QuasiMonteCarlo.sample(N,a,b,SobolSample())
    #return weighted sum (note need to scale based on (b-a))
    return sum(f(X[:,i]) for i in 1:N)/N*prod(b.-a)
end;

Nmax = 10^6
dfQMonte = DataFrame()
dfQMonte.N = 1000:10000:Nmax
for D in 1:10
    dfQMonte[!,"dimension$D"] = [quasi_montecarlo_integration(f_runge_ND,-1*ones(D),ones(D),N) for N in 1000:10000:Nmax]
    dfQMonte[!,"dimension$D"] ./= dfQMonte[end,"dimension$D"] #normalize by last point
end
@df dfQMonte plot(:N,cols(2:11),legend=true)

using BasisMatrices,LinearAlgebra,Roots
σ = 2. #Standard
γ = 2. #Targets Frisch elasticity of 0.5
σ_α = sqrt(0.147) #Taken from HSV

"""
    approximate_household_labor(NlŴ,NT,σ,γ)

Approximates HH policies as a function of log after tax wage and transfers.
"""
function approximate_household_labor(NlŴ,NT,σ_α,σ,γ;scale=5)
    lŴbasis = ChebParams(NlŴ,-scale*σ_α+log(1-.8),scale*σ_α)
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
    budget_residual(τ,T,α,w,hf)

Computes the residual of the HH budget constraint given policy (τ,T)
"""
function budget_residual(τ,T,α,w,hf)
    N = length(α)
    X = [α .+ log(1-τ)  T*ones(N)]
    tax_income = dot(w, hf(X).*exp.(α).*τ)
    return tax_income - T
end;

"""
    government_welfare(τ,T,α,w,σ,γ)

Solves for government welfare given given policy (τ,T)
"""
function government_welfare(τ,T,α,w,Uf)
    N = length(α)
    X = [α .+ log(1-τ)  T*ones(N)]
    return dot(w,Uf(X))
end;

using NLopt

""" 
    find_optimal_policy(N,σ_α,Uf,hf)

Computes the optimal policy given policy fuctions hf and indirect utility Uf
"""
function find_optimal_policy(N,σ_α,Uf,hf)
    ξ,w = gausshermite(N)
    α = sqrt(2).*σ_α.*ξ .- σ_α^2/2
    w ./= sqrt(pi) #remember sqrt(\pi)
    opt = Opt(:LN_COBYLA, 2)
    lower_bounds!(opt, [0., 0.])
    upper_bounds!(opt, [0.8,Inf])
    ftol_rel!(opt,1e-8)

    min_objective!(opt, (x,g)->-government_welfare(x[1],x[2],α,w,Uf))
    equality_constraint!(opt, (x,g) -> -budget_residual(x[1],x[2],α,w,hf))

    minf,minx,ret = NLopt.optimize(opt, [0.3, 0.3])
    if ret == :FTOL_REACHED
        return minx
    end
end
cf,hf,Uf = approximate_household_labor(20,20,σ_α,σ,γ)
find_optimal_policy(30,σ_α,Uf,hf)
@time find_optimal_policy(30,σ_α,Uf,hf)

plot(2:50,[find_optimal_policy(N,σ_α,Uf,hf)[1] for N=2:50],xlabel="Nodes",ylabel="Optimal Tax Rate")

using Distributions
""" 
    find_optimal_policy(N,σ_α,Uf,hf)

Computes the optimal policy given policy fuctions hf and indirect utility Uf
"""
function find_optimal_policy_MC(N,σ_α,Uf,hf)
    alphaDist = Normal(-σ_α^2/2,σ_α)
    α = rand(alphaDist,N)
    w = ones(N)/N #equal weights to compute sample average 
    opt = Opt(:LN_COBYLA, 2)
    lower_bounds!(opt, [0., 0.])
    upper_bounds!(opt, [0.5,Inf])
    ftol_rel!(opt,1e-8)

    min_objective!(opt, (x,g)->-government_welfare(x[1],x[2],α,w,Uf))
    equality_constraint!(opt, (x,g) -> -budget_residual(x[1],x[2],α,w,hf))

    minf,minx,ret = NLopt.optimize(opt, [0.3, 0.3])
    if ret == :FTOL_REACHED
        return minx
    end
end;

plot(2:200,[find_optimal_policy(N,σ_α,Uf,hf)[1] for N=2:200],label="Quadrature",legend=true)
plot!(2:200,[find_optimal_policy_MC(N,σ_α,Uf,hf)[1] for N=2:200],label="Monte Carlo",
    xlabel="Nodes",ylabel="Optimal Tax Rate")