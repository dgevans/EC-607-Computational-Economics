using Gadfly,Parameters,LinearAlgebra


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
end

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
end
    
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
end
lq = pricingLQ(0.99,1.,2.,0.9);
solve_ricatti!(lq)
x,u = simulate_lq(lq,[1.,0.],20)

plot(layer(y=x[1,:],Geom.line,color=["Target Price"]),
     layer(y=x[2,:],Geom.line,color=["Price"]),Guide.xlabel("Time"))


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

lq = pricingLQ(1.,1.,2.,0.9);

@unpack A,B,R,Q = lq
n = size(A,1)
L = [I B*inv(Q)*B'; 0*R A']
N = [A 0*A; -R I]
M = L\N

F = schur(M)
ordschur!(F,abs.(F.values) .< 1)
