using StatsPlots,Parameters,LinearAlgebra
default(linewidth=2,legend=false)
mutable struct LQProblem
    #Objective
    β::Float64
    R::Matrix{Float64}
    Q::Matrix{Float64}

    #Constraint
    A::Matrix{Float64}
    B::Matrix{Float64}

    #Solution
    P::Matrix{Float64}
    F::Matrix{Float64}
end;

"""
    iterate_ricatti(lq::LQProblem,P)    

Iterates on the Ricatti euqation given value function x'Px
"""
function iterate_ricatti(lq::LQProblem,P)
    @unpack β,R,Q,A,B = lq

    return R + β*A'*P*A - β^2*A'*P*B*inv(Q+B'*P*B)*B'*P*A
end

"""
    sovle_ricatti(lq)

Solves the Ricatti equation 
"""
function solve_ricatti!(lq)
    @unpack β,R,Q,A,B = lq
    P = copy(R)
    diff = 1.
    while diff>1e-8
        P′ =  iterate_ricatti(lq,P)
        diff = norm(P′-P,Inf)
        P = P′
    end

    lq.P = P
    lq.F = β*inv(Q+B'*P*B)*B'*P*A
end;

"""
    simulate_lq(lq,x0,T)

Simulates a solution to the LQ problem assuming lq.P and lq.F 
are already given.
"""
function simulate_lq(lq,x0,T)
    @unpack F,A,B = lq
    x = zeros(length(x0),T+1)
    u = zeros(size(F,1),T)

    x[:,1] = x0
    for t in 1:T
        u[:,t] = -F*x[:,t]
        x[:,t+1] = A*x[:,t] + B*u[:,t]
    end

    return (x=x,u=u)
end;

"""
    pricingLQ(β,γ_1,γ_2,ρ)

Setup the LQ problem of a firm.
"""
function pricingLQ(β,γ_1,γ_2,ρ)
    R = γ_1 .* [1 -1;-1 1]
    Q = γ_2 .* ones(1,1)

    A = [ρ 0;0 1]
    B = [0. ;1. ][:,:]

    return LQProblem(β,R,Q,A,B,zeros(1,1),zeros(1,1))
end
lq = pricingLQ(1.,1.,2.,0.1);

solve_ricatti!(lq)
x,u = simulate_lq(lq,[1.,0.],20)
plot(x[1,:],label="Target Price",legend=true)
plot!(x[2,:],label="Price",xlabel="Time",ylabel="Price")

"""
    simulate_costate(lq,x0,μ0,T)

Simulates the path of variables using the costate sytem.
Assumes no-discounting
"""
function simulate_costate(lq,x0,μ0,T)
    @unpack A,B,R,Q = lq
    L = [I B*inv(Q)*B'; 0*R A']
    N = [A 0*A; -R I]
    M = L\N

    y = zeros(2*length(x0),T+1)
    y[:,1] = [x0;μ0]
    for t in 1:T
        y[:,t+1] = M*y[:,t]
    end

    return y
end;

x0 = [1.,0.]
solve_ricatti!(lq)
y_low = simulate_costate(lq,x0,lq.P*x0 - 0.001*ones(2),20)
y_correct = simulate_costate(lq,x0,lq.P*x0,20)
y_high = simulate_costate(lq,x0,lq.P*x0 + 0.001*ones(2),20)
plot(y_low[2,:],label="Low Initial Costate",legend=true)
plot!(y_correct[2,:],label="Correct Initial Costate")
plot!(y_high[2,:],label="High Initial Costate",
     xlabel="Time",ylabel="Price")

@unpack A,B,R,Q = lq
L = [I B*inv(Q)*B'; 0*R A']
N = [A 0*A; -R I]
M = L\N
M*[zeros(2,2) -I;I zeros(2,2)]*M'

using LinearAlgebra
F = schur(M)
F.Z

F.T

println(eigvals(M))

"""
    solve_ricatti_stable!(lq)

Solves the Ricatti equation by finding the unique stable path.
"""
function solve_ricatti_stable!(lq)
    @unpack A,B,R,Q = lq
    n = size(A,1)
    L = [I B*inv(Q)*B'; 0*R A']
    N = [A 0*A; -R I]
    M = L\N

    F = schur(M)
    ordschur!(F,F.values .< 1) #stack stable eigenvalues first

    Z = F.Z
    Z_11 = Z[1:n,1:n]
    Z_21 = Z[n+1:end,1:n]
    lq.P = P =  Z_21*inv(Z_11)
    lq.F = inv(Q+B'*P*B)*B'*P*A
end;

solve_ricatti!(lq)
lq.P

solve_ricatti_stable!(lq)
lq.P

using Parameters
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
    compute_steadystate(model::RBCModel)

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

using ForwardDiff
model = RBCModel()
x̄,ȳ,ε̄ = compute_steadystate(model)
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
end;
compute_dx!(model,F_RBC,x̄,ȳ,ε̄)
#check: should be all 0
F̄_x_ + F̄_x*model.ḡ_x+F̄_y*model.h̄_x+F̄_y′*model.h̄_x*model.ḡ_x

"""
    compute_dε!(model,F,x̄,ȳ,ε̄)

Computes derivatives w.r.t. ε of law of motion g and 
and policy rules h.
"""
function compute_dε!(model,F,x̄,ȳ,ε̄)
    @unpack h̄_x = model
    nx = length(x̄)
    ny = length(ȳ)
    F̄_ε = ForwardDiff.jacobian(ε->F(model,x̄,x̄,ȳ,ȳ,ε),ε̄)
    F̄_x = ForwardDiff.jacobian(x->F(model,x̄,x,ȳ,ȳ,ε̄),x̄)
    F̄_y = ForwardDiff.jacobian(y->F(model,x̄,x̄,y,ȳ,ε̄),ȳ)
    F̄_y′ = ForwardDiff.jacobian(y′->F(model,x̄,x̄,ȳ,y′,ε̄),ȳ)

    #G and A matrices
    dw_ε = -inv([F̄_x+F̄_y′*h̄_x F̄_y])*F̄_ε
    model.ḡ_ε = dw_ε[1:nx,:]
    model.h̄_ε = dw_ε[nx+1:end,:]
end
compute_dε!(model,F_RBC,x̄,ȳ,ε̄);

using DataFrames
"""
    compute_irf_RBC(model,T)

Computes the IRF to a 1 standard deviation shock to productivity
"""
function compute_irf_RBC(model,T)
    @unpack ḡ_x,ḡ_ε,h̄_x,h̄_ε,σ_ε = model
    nx = size(ḡ_x,1)
    ny = size(h̄_x,1)

    x = zeros(nx,T+1)
    y = zeros(ny,T)
    y[:,1] = h̄_ε*σ_ε
    x[:,2] = ḡ_ε*σ_ε
    for t in 2:T
        y[:,t] = h̄_x*x[:,t]
        x[:,t+1] = ḡ_x*x[:,t]
    end

    return DataFrame(t=1:T,c=y[1,:],R=y[2,:],a_=x[1,1:T],k_=x[2,1:T])
end;

df = compute_irf_RBC(model,50)
@df df plot(:t,[:R,:a_,:c,:k_],layout=4,title=["R" "a_" "c" "k_"])

"""
    simulate_RBC(model,T)

Simulates the path of endogeneous variables for T periods
"""
function simulate_RBC(model,T)
    @unpack ḡ_x,ḡ_ε,h̄_x,h̄_ε,σ_ε = model
    nx = size(ḡ_x,1)
    ny = size(h̄_x,1)

    x = zeros(nx,T+1)
    y = zeros(ny,T)
    ε = randn(T)
    y[:,1] = h̄_ε*σ_ε*ε[1]
    x[:,2] = ḡ_ε*σ_ε*ε[1]
    for t in 2:T
        y[:,t] = h̄_x*x[:,t] + h̄_ε*σ_ε*ε[t]
        x[:,t+1] = ḡ_x*x[:,t] + ḡ_ε*σ_ε*ε[t]
    end

    return DataFrame(t=1:T,c=y[1,:],R=y[2,:],a_=x[1,1:T],k_=x[2,1:T],ε=ε)
end;

df = simulate_RBC(model,200)
@df df plot(:t,[:R,:a_,:c,:k_],layout=4,title=["R" "a_" "c" "k_"])

"""
    linearize_sequence(model,F,x̄,ȳ,ε̄,T=100)

Linearizes the model using sequence space rather than schur decomposition
"""
function linearize_sequence(model,F,x̄,ȳ,ε̄,T=100)
    nx = length(x̄)
    ny = length(ȳ)
    F̄_x_ = ForwardDiff.jacobian(x_->F(model,x_,x̄,ȳ,ȳ,ε̄),x̄)
    F̄_x = ForwardDiff.jacobian(x->F(model,x̄,x,ȳ,ȳ,ε̄),x̄)
    F̄_y = ForwardDiff.jacobian(y->F(model,x̄,x̄,y,ȳ,ε̄),ȳ)
    F̄_y′ = ForwardDiff.jacobian(y′->F(model,x̄,x̄,ȳ,y′,ε̄),ȳ)
    F̄_ε = ForwardDiff.jacobian(ε->F(model,x̄,x̄,ȳ,ȳ,ε),ε̄)

    J = zeros(nx+ny,T,nx+ny,T)
    A = zeros(nx+ny,T)
    #setup for t=0
    A[:,1] = F̄_ε
    J[:,1,:,1] = [F̄_x F̄_y]
    J[:,1,:,2] =  [zeros(nx+ny,nx) F̄_y′]
    for t in 2:T-1
        J[:,t,:,t-1] =[F̄_x_ zeros(nx+ny,ny)] 
        J[:,t,:,t] = [F̄_x F̄_y]
        J[:,t,:,t+1] = [zeros(nx+ny,nx) F̄_y′]
    end
    J[:,T,:,T-1] =[F̄_x_ zeros(nx+ny,ny)] 
    J[:,T,:,T] = [F̄_x F̄_y]

    J = reshape(J,(nx+ny)*(T),:)
    A = A[:]

    return reshape(-inv(J)*A,:,T)
end;

df_irf = compute_irf_RBC(model,100)
w = linearize_sequence(model,F_RBC,x̄,ȳ,ε̄)
@df df_irf scatter(:t,:c,label="Standard Linearization",legend=true)
scatter!(w[3,:]*model.σ_ε,label="Sequence Linearization")

df = compute_irf_RBC(model,25)
w = linearize_sequence(model,F_RBC,x̄,ȳ,ε̄,25)
@df df_irf scatter(:t,:c,label="Standard Linearization",legend=true)
scatter!(w[3,:]*model.σ_ε,label="Sequence Linearization")

"""
    simulate_RBC_sequence(model,T)

Simulates the path of endogeneous variables for T periods, using
sequence linearization method
"""
function simulate_RBC_sequence(model,T,Tlin)
    @unpack σ_ε = model
    x̄,ȳ,ε̄ = compute_steadystate(model)
    w_ε = linearize_sequence(model,F_RBC,x̄,ȳ,ε̄,Tlin+1)
    nx = length(x̄)
    ny = length(ȳ)
    x_ε = w_ε[1:nx,:]
    y_ε = w_ε[nx+1:end,:]
    x = zeros(nx,T+Tlin+1)
    y = zeros(ny,T+Tlin)
    ε = randn(T)
    for t in 1:T
        y[:,t:t+Tlin] += y_ε*σ_ε*ε[t]
        x[:,t+1:t+Tlin+1] += x_ε*σ_ε*ε[t]
    end

    return DataFrame(t=1:T,c=y[1,1:T],R=y[2,1:T],a_=x[1,1:T],k_=x[2,1:T],ε=ε)
end;

using Random
Random.seed!(21323)
df = simulate_RBC(model,200)
Random.seed!(21323)
df2 = simulate_RBC_sequence(model,200,100)
#Compare w/ same random seed
@df df  plot(:t,:c,label="Linearization",legend=true)
@df df2 plot!(:t,:c,label="Sequence Linearization",xlabel="Time",ylabel="Consumption")

using MATLAB
mat"cd 'Lectures/Perturbation Theory'" #Mod file has to be in current directory
mat"dynare RBC.mod" #As easy as that

c_dyn = @mget c_e;

@df df_irf plot(:c,label="Linearization",legend=true)
plot!(c_dyn,label="Dynare",ylabel="c")

mat"dynare RBC2.mod" #As easy as that

c_dyn = @mget c_e;
@df df_irf plot(:c,label="Linearization",legend=true)
plot!(c_dyn,label="Dynare",ylabel="c")