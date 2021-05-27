using Gadfly,ForwardDiff,Parameters

f(a) = a^2

df(a) = ForwardDiff.derivative(f,a)
d2f(a) = ForwardDiff.derivative(df,a)


g(x,a) = [sin(a*x[1]),exp(x[2]+sin(a*x[1]))]

Dg(x) = ForwardDiff.jacobian(x->g(x,2),x)




@with_kw mutable struct RBCModel
    α::Float64 = 0.3 #Capital Share of Output
    β::Float64 = 0.96 #Discount Factor
    δ::Float64 = 0.1 #Depreciation Rate
    σ::Float64 = 2.  #1/IES
    ρ::Float64 = 0.85 #persistence of log productivity
    σ_ε::Float64 = 0.015 #innovation to log productivity

    #Store first order Approximation
    ḡ_x::Matrix{Float64} = zeros(0,0)
    ḡ_ε::Matrix{Float64} = zeros(0,0)

    h̄_x::Matrix{Float64} = zeros(0,0)
    h̄_ε::Matrix{Float64} = zeros(0,0)
end;

"""
    F_RBC(model::RBCModel,x_,x,y,y′)

Equations governing the RBC model
"""
function F_RBC(model::RBCModel,x_,x,y,y′,ε)
    @unpack α,β,δ,σ,ρ = model
    #unpack variables
    a_,k_ = x_
    a,k = x
    c,R = y
    c′,R′ = y′
    A = exp(a)
    #return equations
    return[ c + k - A*k_^α - (1-δ)*k_,
            R - α*A*k_^(α-1) - 1 + δ,
            β*R′*(c′)^(-σ) - c^(-σ),
            a - ρ*a_ - ε[1]] #allowing ε to be a vector
end;


"""
    compute_steadystate(para::RBCModel)

Computes steady state given parameters stored in para
"""
function compute_steadystate(model::RBCModel)
    @unpack α,β,δ,σ = model
    R̄ = 1/β
    k̄ =((R̄ - 1 + δ)/(α))^(1/(α-1))

    c̄ = k̄^α - δ*k̄

    x̄ = [0,k̄]
    ȳ = [c̄,R̄]
    return x̄,ȳ,zeros(1)
end;

model = RBCModel()
x̄,ȳ,ε̄ = compute_steadystate(model)

using ForwardDiff
F̄_ε = ForwardDiff.jacobian(ε->F_RBC(model,x̄,x̄,ȳ,ȳ,ε),ε̄)
F̄_x_ = ForwardDiff.jacobian(x_->F_RBC(model,x_,x̄,ȳ,ȳ,ε̄),x̄)
F̄_x = ForwardDiff.jacobian(x->F_RBC(model,x̄,x,ȳ,ȳ,ε̄),x̄)
F̄_y = ForwardDiff.jacobian(y->F_RBC(model,x̄,x̄,y,ȳ,ε̄),ȳ)
F̄_y′ = ForwardDiff.jacobian(y′->F_RBC(model,x̄,x̄,ȳ,y′,ε̄),ȳ)

"""
    compute_dx(model,F,x̄,ȳ,ε̄)

Computes derivatives w.r.t. x of law of motion g and 
and policy rules h.
"""
function compute_dx!(model,F,x̄,ȳ,ε̄)
    nx = length(x̄)
    ny = length(ȳ)
    F̄_x_ = ForwardDiff.jacobian(x_->F(model,x_,x̄,ȳ,ȳ,ε̄),x̄)
    F̄_x = ForwardDiff.jacobian(x->F(model,x̄,x,ȳ,ȳ,ε̄),x̄)
    F̄_y = ForwardDiff.jacobian(y->F(model,x̄,x̄,y,ȳ,ε̄),ȳ)
    F̄_y′ = ForwardDiff.jacobian(y′->F(model,x̄,x̄,ȳ,y′,ε̄),ȳ)

    #G and A matrices
    G = [F̄_x F̄_y′]
    A = -[F̄_x_ F̄_y]

    F = schur(A,G) #generalized schur decomposition
    λ = abs.(F.α./F.β) #compute ratio of eigenvalues
    if sum(λ .> 1) > length(ȳ)
        error("Equlibrium does not exist")
    end
    if sum(λ .> 1) < length(ȳ)
        error("Equilibrium is not unique")
    end
    ordschur!(F,λ.<=1) #Put stable eigenvectors first
    Zxθ = F.Z[1:nx,1:nx]
    Zyθ = F.Z[nx+1:end,1:nx]
    Sθθ = F.S[1:nx,1:nx]
    Tθθ = F.T[1:nx,1:nx]
    
    model.h̄_x = Zyθ*inv(Zxθ)
    model.ḡ_x = Zxθ*inv(Tθθ)*Sθθ*inv(Zxθ)
end

compute_dx!(model,F_RBC,x̄,ȳ,ε̄)
F̄_x_ + F̄_x*model.ḡ_x+F̄_y*model.h̄_x+F̄_y′*model.h̄_x*model.ḡ_x