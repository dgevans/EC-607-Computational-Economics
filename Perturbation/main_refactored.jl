include("AiyagariModel_CA.jl")
include("ZerothOrderApproximation.jl")
include("FirstOrderApproximationRefactored.jl")

# Choose Parameters for the run
σ = 5 #risk aversion
ϕ = 35.0 #adjustment cost
α = 0.36 # capital share

ρ_Θ = 0.8 #persistence of agg TFP
Σ_Θ= 0.014^2*ones(1,1)

Nb = 120
Ib = 1000

Nϵ = 7

T=400 #truncation length


AM = AiyagariModelCA()
AM.τ_θ= 0.
AM.ϕ = ϕ
AM.ρ_Θ .= ρ_Θ
AM.σ = σ
AM.α = α
AM.Nb = Nb
AM.Ib = Ib
AM.Σ_Θ =Σ_Θ
AM.Nϵ = Nϵ
#setup_approximation!(AM); #Solve for the steady state and setup basis matrices

save_policy_functions!(AM);
save_agg!(AM);
save_H!(AM);

function construct_xfs(AM::AiyagariModelCA)
    @unpack λf,bf,vf,b̂grid,b̄grid= AM #then unpack equilibrium objects
    ẑgrid =b̂grid
    z̄grid = b̄grid
    xf=[bf,λf,vf]
    ix=Dict(:b=>1,:λ=>2,:v=>3)
    iz=ix[:b]
    return xf,ix,iz,ẑgrid,z̄grid

end

function construct_X̄s(AM::AiyagariModelCA)
    @unpack R̄,W̄,T̄,ℐ,C̄,Ȳ,Iv,K̄,q̄ = AM 
    X̄=[R̄,W̄,T̄,ℐ,C̄,Ȳ,Iv,K̄,q̄ ]
    iR,iW,iT,iℐ,iC,iY,iIv,iK,iq = 1,2,3,4,5,6,7,8,9
    iX=Dict(:R̄=>iR,:W̄=>iW,:T̄=>iT,:ℐ=>iℐ,:C̄=>iC,:Ȳ=>iY,:Iv=>iIv,:K̄=>iK,:q̄=>iq)
    iQ=[iR, iW, iT]
    iQ_=[iK, iq]
    return X̄,iX,iQ,iQ_
end

para=ModelParams(b_cutoff = AM.b_cutoff, b̲ = AM.b̲, β = AM.β, σ = AM.σ, ϵ = AM.ϵ, α = AM.α, δ = AM.δ, N̄ = AM.N̄, ϕ = AM.ϕ, τ_θ = AM.τ_θ)


function F(para::ModelParams,θ,z_,x,X,x′)
    @unpack b_cutoff,β,σ,ϵ,b̲ = para
    #unpack variables
    b,λ,v = x
    _,Eλ′,Ev′ = x′
    R,W,T = X

    c = (λ/R)^(-1/σ)


    ret = [R*z_+W*ϵ[θ]+T-c-b,
           v - c^(1-σ)/(1-σ) - β*Ev′,
           β*Eλ′-λ/R]
    if z_ < b_cutoff[θ]
        ret[3] = b̲-b
    end
    return ret
end

function G(para::ModelParams,Ix,X_,X,Θ)
    @unpack α,δ,N̄,ϕ,τ_θ = para
    qK,_,Iv = Ix
    R,W,T,ℐ,C,Y,V,K,q = X
    K_,q_ = X_
    #now perform operations
    rK = α*Θ[1]*K_^(α-1)*N̄^(1-α)
    ϕK = ℐ +  0.5*ϕ*(ℐ-δ)^2
    τ = τ_θ*(Θ[1]-1.0)
    return [qK - q*K,#
            (q*(1 - δ +ℐ) + rK - ϕK)/q_ - R,#
            (1-α)Θ[1]*K_^α*N̄^(-α)*(1-τ) - W, #
            K - (1 - δ + ℐ)*K_,#
            Y - C - ϕK*K_,#
            q - 1 - ϕ*(ℐ-δ),#
            Y - Θ[1]*K_^(α)*N̄^(1-α),
            V - Iv,#
            T - τ*(1-α)*Θ[1]*K_^α*N̄^(-α)]#
end

inputs=Inputs()
#A
inputs.xf,inputs.ix,inputs.iz,inputs.ẑgrid,inputs.z̄grid=construct_xfs(AM);
#B
inputs.X̄,inputs.iX,inputs.iQ,inputs.iQ_ = construct_X̄s(AM);
#C
inputs.F = (θ,z_,x,X,x′)->F(para,θ,z_,x,X,x′)
inputs.G = (Ix,X_,X,Θ)->G(para,Ix,X_,X,Θ)
#D
inputs.ω̄ ,inputs.πθ =  AM.ω̄ ,AM.πϵ;
inputs.Θ̄,inputs.ρ_Θ,inputs.Σ_Θ = ones(1)*AM.Θ̄,AM.ρ_Θ,AM.Σ_Θ  

ZO=ZerothOrderApproximation(inputs)
@time computeDerivativesF!(ZO,inputs)
ZO.dF

@time computeDerivativesG!(ZO,inputs)
ZO.dG

T=400
include("FirstOrderApproximationRefactored.jl")
FO = FirstOrderApproximation(ZO,T)#define the FirstOrderApproximation object
@time compute_Θ_derivatives!(FO,inputs)
plot(FO.X̄_Z[inputs.iX[:K̄],:])

