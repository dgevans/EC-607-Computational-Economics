using Parameters,LinearAlgebra

@with_kw mutable struct NKParameters
    #parameters
    α::Float64 = 0.3
    β::Float64 = 0.99
    θ::Float64 = 0.45
    σ::Float64 = 2.
    φ::Float64 = 2.
    ϵ::Float64 = 6.
    ρ_ν::Float64 = 0.
    #Taylor Rule
    ϕ_π::Float64 = 1.5
    ϕ_y::Float64 = 0.
end

function constructA(para::NKParameters)
    @unpack α,β,θ,σ,φ,ϵ,ϕ_π,ϕ_y,ρ_ν = para
    #constants
    bigθ = (1-α)/(1-α+α*ϵ)
    λ = ((1-θ)*(1-β*θ)/θ )* bigθ
    κ = λ*(σ+(φ+α)/(1-α))

    A = [ρ_ν             0                 0;
        0                1/β          -κ/β;
        1/σ        (ϕ_π-1/β)/σ  (ϕ_y+κ/β)/σ+1;
        ]
    return A
end

function checkdeterminacy(para::NKParameters)
    A = constructA(para)
    F = schur(A)
    return sum(norm.(F.values).>1)
end

function findpolicy(para::NKParameters)
    A = constructA(para)
    F = schur(A)
    ordschur!(F,norm.(F.values).<= 1.)

    Z = F.Z
    T = F.T

    Zxθ= Z[1:1,1:1]
    Zyθ= Z[2:end,1:1]
    return Zyθ*inv(Zxθ)
end

para = NKParameters()
checkdeterminacy(para)
findpolicy(para)


para.ϕ_π = 1.001
checkdeterminacy(para)

