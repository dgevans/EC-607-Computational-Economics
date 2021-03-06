using Plots
default(linewidth=2,legend=false,margin=10Plots.mm)

using Optim #Our optimization libarary
f_univ(x) = 2x^2+3x+1
plot(f_univ,-2.,1)

#Finds global minimum
res = optimize(f_univ,-2.0,1.0,GoldenSection())
Optim.minimizer(res)

#Lower bound binds
res = optimize(f_univ,-0.5,1.0,GoldenSection())
Optim.minimizer(res)

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
    household_labor(??,??,T,??,??)

Solves for HH labor choice given policy and preferences
"""
function household_labor(??,??,T,??,??)
    W?? = (1-??)*exp(??) #after tax wages
    res(h) = (W??*h+T)^(-??)*W??-h^??
    min_h = max(0,(.0000001-T)/(1-??)*exp(??)) #ensures c>.0001
    h = fzero(res,min_h,20000.) #find hours that solve HH problem
    c = W??*h+T
    U = c^(1-??)/(1-??)-h^(1+??)/(1+??)
    return c,h,U
end

"""
    budget_residual(??,T,??vec,??,??)

Computes the residual of the HH budget constraint given policy (??,T).
??vec contains the vector of ?? values for each agent
"""
function budget_residual(??,T,??vec,??,??)
    tax_income = 0.
    N = length(??vec)
    for i in 1:N
        c,h,U = household_labor(??vec[i],??,T,??,??)
        tax_income += ??*h*exp(??vec[i])
    end
    return tax_income/N - T
end

"""
    government_welfare(??,??vec,??,??)

Solves for government welfare given tax rate ??
"""
function government_welfare(??,??vec,??,??)
    f(T) = budget_residual(??,T,??vec,??,??)
    T =  fzero(f,0.) #Find transfers that balance budget
    
    welfare = 0.
    N = length(??vec)
    for i in 1:N
        #compute HH welfare given tax rate
        c,h,U = household_labor(??vec[i],??,T,??,??)
        welfare += U #Aggregate welfare
    end
    return welfare/N
end

using Distributions
?? = 2. #Standard
?? = 2. #Targets Frisch elasticity of 0.5
??_?? = sqrt(0.147) #Taken from HSV
N = 1000
alphaDist = Normal(-??_??^2/2,??_??)
??vec = rand(alphaDist,N);

plot(??->government_welfare(??,??vec,??,??),0.,0.8,ylabel="Welfare",xlabel="Tax Rate")

@time minx_optim = Optim.optimize(??->-government_welfare(??,??vec,??,??),0.,0.8)
println(minx_optim)

"""
    government_welfare(??,T,??vec,??,??)

Solves for government welfare given tax rate ??
"""
function government_welfare(??,T,??vec,??,??)
    welfare = 0.
    N = length(??vec)
    for i in 1:N
        #compute HH welfare given tax rate
        c,h,U = household_labor(??vec[i],??,T,??,??)
        welfare += U #Aggregate welfare
    end
    return welfare/N
end

opt = Opt(:LN_COBYLA, 2)

lower_bounds!(opt, [0., -1.]) #x[1] is tau, x[2] is T
upper_bounds!(opt, [0.8,Inf])
ftol_rel!(opt,1e-8)

min_objective!(opt, (x,g)->-government_welfare(x[1],x[2],??vec,??,??))
equality_constraint!(opt, (x,g) -> -budget_residual(x[1],x[2],??vec,??,??))

@time (minf,minx_nlopt,ret) = NLopt.optimize(opt, [0.3, 0.3])
println(minx_nlopt[1])