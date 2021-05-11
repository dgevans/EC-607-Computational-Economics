
using Roots,Optim
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

#Calibration
using Distributions
σ = 2. #Standard
γ = 2. #Targets Frisch elasticity of 0.5
σ_α = sqrt(0.147) #Taken from HSV
N = 1000
alphaDist = Normal(-σ_α^2/2,σ_α)
αvec = rand(alphaDist,N);


"""
    budget_residual(τ,T,αvec,σ,γ)

Computes the residual of the HH budget constraint given policy (τ,T)
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

Optim.optimize(τ->-government_welfare(τ,αvec,σ,γ),0.,0.8)

plot(τ->government_welfare(τ,αvec,σ,γ),0.,0.8)


f = (x,g) -> budget_residual(x[1],x[2],αvec,σ,γ)


