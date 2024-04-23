using Plots
default(linewidth=2,legend=false,margin=10Plots.mm)

using Optim #Our optimization libarary
f_univ(x) = 2x^2+3x+1
plot(f_univ,-2.,1)

#Finds global minimum
ret = optimize(f_univ,-2.0,1.0,GoldenSection())
Optim.minimizer(ret)

#Lower bound binds
ret = optimize(f_univ,-0.5,1.0,GoldenSection())
Optim.minimizer(ret)

optimize(f_univ,-2.0,1.0,Brent()) #run once to precompile
@time res = optimize(f_univ,-2.0,1.0,Brent());

@time res = optimize(f_univ,-2.0,1.0,GoldenSection());

#rosenbrock function
f_ros(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
optimize(f_ros,10*ones(2),NelderMead())

#uses by default AffineSimplexer()
Optim.simplexer(Optim.AffineSimplexer(),[0.,1.]);

#Simulated Annealing
result = optimize(f_ros,zeros(2),SimulatedAnnealing(),Optim.Options(iterations=10^7)) #very rough right now

#Obtaining solutions
println("Value of Function at Minimum:$(result.minimum)")
println("Point that Acheives Minimum:$(result.minimizer)")

optimize(f_ros,zeros(2),BFGS())

@time optimize(f_ros,zeros(2),NelderMead())
@time optimize(f_ros,zeros(2),SimulatedAnnealing(),Optim.Options(iterations=10^7))
@time optimize(f_ros,zeros(2),BFGS());

using NLopt
function myfunc!(x::Vector, grad::Vector)
    if length(grad) > 0
        grad[1] = 0
        grad[2] = 0.5/sqrt(x[2])
    end

    sqrt(x[2])
end

function myconstraint!(x::Vector, grad::Vector, a, b)
    if length(grad) > 0
        grad[1] = 3a * (a*x[1] + b)^2
        grad[2] = -1
    end
    (a*x[1] + b)^3 - x[2]
end

opt = Opt(:LD_MMA, 2)
lower_bounds!(opt, [-Inf, 0.])
ftol_rel!(opt,1e-6)

min_objective!(opt, myfunc!)
inequality_constraint!(opt, (x,g) -> myconstraint!(x,g,2,0))
inequality_constraint!(opt, (x,g) -> myconstraint!(x,g,-1,1))

(minf,minx,ret) = NLopt.optimize(opt, [3., 1.])
println("got $minf at $minx after $count iterations (returned $ret)")

opt = Opt(:LD_SLSQP, 2)
lower_bounds!(opt, [-Inf, 0.])
ftol_rel!(opt,1e-4)

min_objective!(opt, myfunc!)
inequality_constraint!(opt, (x,g) -> myconstraint!(x,g,2,0))
inequality_constraint!(opt, (x,g) -> myconstraint!(x,g,-1,1))

(minf,minx,ret) = NLopt.optimize(opt, [3., 1.])
println("got $minf at $minx after $count iterations (returned $ret)")

using Roots
"""
    household_labor(α,τ,T,σ,γ)

Solves for HH labor choice given policy and preferences
"""
function household_labor(α,τ,T,σ,γ)
    Ŵ = (1-τ)*exp(α) #after tax wages
    res(h) = (Ŵ*h+T)^(-σ)*Ŵ-h^γ
    min_h = max(0,(.0000001-T)/(1-τ)*exp(α)) #ensures c>.0001
    h = fzero(res,min_h,20000.) #find hours that solve HH problem
    c = Ŵ*h+T
    U = c^(1-σ)/(1-σ)-h^(1+γ)/(1+γ)
    return c,h,U
end

"""
    budget_residual(τ,T,αvec,σ,γ)

Computes the residual of the HH budget constraint given policy (τ,T).
αvec contains the vector of α values for each agent
"""
function budget_residual(τ,T,αvec,σ,γ)
    tax_income = 0.
    N = length(αvec)
    for i in 1:N
        c,h,U = household_labor(αvec[i],τ,T,σ,γ)
        tax_income += τ*h*exp(αvec[i])
    end
    return tax_income/N - T
end

"""
    government_welfare(τ,αvec,σ,γ)

Solves for government welfare given tax rate τ
"""
function government_welfare(τ,αvec,σ,γ)
    f(T) = budget_residual(τ,T,αvec,σ,γ)
    T =  fzero(f,0.) #Find transfers that balance budget
    
    welfare = 0.
    N = length(αvec)
    for i in 1:N
        #compute HH welfare given tax rate
        c,h,U = household_labor(αvec[i],τ,T,σ,γ)
        welfare += U #Aggregate welfare
    end
    return welfare/N
end

using Distributions
σ = 2. #Standard
γ = 2. #Targets Frisch elasticity of 0.5
σ_α = sqrt(0.147) #Taken from HSV
N = 1000
alphaDist = Normal(-σ_α^2/2,σ_α)
αvec = rand(alphaDist,N);

plot(τ->government_welfare(τ,αvec,σ,γ),0.,0.8,ylabel="Welfare",xlabel="Tax Rate")

@time minx_optim = Optim.optimize(τ->-government_welfare(τ,αvec,σ,γ),0.,0.8)
println(minx_optim)

"""
    government_welfare(τ,T,αvec,σ,γ)

Solves for government welfare given tax rate τ
"""
function government_welfare(τ,T,αvec,σ,γ)
    welfare = 0.
    N = length(αvec)
    for i in 1:N
        #compute HH welfare given tax rate
        c,h,U = household_labor(αvec[i],τ,T,σ,γ)
        welfare += U #Aggregate welfare
    end
    return welfare/N
end

opt = Opt(:LN_COBYLA, 2)

lower_bounds!(opt, [0., -1.]) #x[1] is tau, x[2] is T
upper_bounds!(opt, [0.8,Inf])
ftol_rel!(opt,1e-8)

min_objective!(opt, (x,g)->-government_welfare(x[1],x[2],αvec,σ,γ))
equality_constraint!(opt, (x,g) -> -budget_residual(x[1],x[2],αvec,σ,γ))

@time (minf,minx_nlopt,ret) = NLopt.optimize(opt, [0.3, 0.3])
println(minx_nlopt[1])