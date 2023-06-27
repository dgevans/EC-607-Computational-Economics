using Parameters,BasisMatrices,SparseArrays,SuiteSparse,TensorOperations
include("utilities.jl")
#Some Helper functions
import Base./,Base.*
"""
    /(A::Array{Float64,3},B::SparseMatrixCSC{Float64,Int64})

Apply the inverse to the last dimension of a 3 dimensional array
"""
function /(A::Array{Float64,3},B::SuiteSparse.UMFPACK.UmfpackLU{Float64, Int64})
    ret = similar(A)
    n = size(ret,1)
    for i in 1:n
        ret[i,:,:] .= (B'\view(A,i,:,:)')' 
        #ret[i,:,:] .= A[i,:,:]/B
    end
    return ret
end


#function *(A::Matrix{Float64},B::Array{Float64,3})
#    return @tensor C[i,k,l] := A[i,j]*B[j,k,l] 
#end

function *(A::Array{Float64,3},B::SparseMatrixCSC{Float64, Int64})
    k,m,n = size(A)
    return reshape(reshape(A,:,n)*B ,k,m,:) 
end

#function *(A::Matrix{Float64},B::Array{Float64,3})
#    return @tensor C[i,k,l] := A[i,j]*B[j,k,l] 
#end

function *(A::SparseMatrixCSC{T, Int64},B::Array{Float64,3}) where {T<:Real}
    k,m,n = size(B)
    return reshape(A*reshape(B,k,:),:,m,n) 
end

"""
FirstOrderApproximation{Model}

Holds all the objects necessary for a first order approximation of a 
given Model.  Will assume F_x(M::Model,z) etc. exists.
"""
@with_kw mutable struct FirstOrderApproximation
    #M::Model #holds objects that we care about like H
    ZO::ZerothOrderApproximation
    T::Int #Length of IRF

    #Derivative direction
    Δ_0::Vector{Float64} = zeros(1) #Distribution direction (will assume 0 for now)
    X_0::Vector{Float64} = zeros(1)
    Θ_0::Vector{Float64} = zeros(1) #Θ direction
    
    #Terms for Lemma 2
    f::Vector{Matrix{Float64}} = Vector{Matrix{Float64}}(undef,1) 
    x::Vector{Array{Float64,3}} = Vector{Array{Float64,3}}(undef,1)
    
    #Terms for Lemma 4
    a::Array{Float64,3} = zeros(1,1,1) #a terms from paper
    z::Array{Float64,3} = zeros(1,1,1) #a terms from paper
    L::SparseMatrixCSC{Float64,Int64} = spzeros(1,1) #A operator
    M::SparseMatrixCSC{Float64, Int64} = spzeros(1,1) #M operator
    
    #Terms for Corollary 2
    IL::Array{Float64, 3} = zeros(1,1,1)
    ILM::Array{Float64, 3} = zeros(1,1,1)
    E::Array{Float64,3} = zeros(1,1,1) #expectations operators
    J::Array{Float64,4} = zeros(1,1,1,1)

    #Terms for Proposition 1
    BB::SparseMatrixCSC{Float64, Int64} = spzeros(1,1)
    

    #Outputs
    Ω̄_Z::Matrix{Float64} =  zeros(1,1)
    x̄_Z::Vector{Matrix{Float64}} =  Vector{Matrix{Float64}}(undef,1)
    X̄_Z::Matrix{Float64} = zeros(1,1)


    Ω̄_Θ::Vector{Matrix{Float64}} =  Vector{Matrix{Float64}}(undef,1)
    x̄_Θ::Vector{Vector{Matrix{Float64}}} =  Vector{Vector{Matrix{Float64}}}(undef,1)
    X̄_Θ::Vector{Matrix{Float64}} = Vector{Matrix{Float64}}(undef,1)
end


# GOOD? 
"""
    FirstOrder(M,T)

Constructs a first order approximation object given a model
"""
function FirstOrderApproximation(ZO::ZerothOrderApproximation,T)
    N = length(ZO.ẑ)
    approx = FirstOrderApproximation(ZO=ZO,T=T)
    #create vectors of appropriate length
    approx.f = Vector{Matrix{Float64}}(undef,N)
    approx.x = Vector{Array{Float64,3}}(undef,T)
    #approx.nx = ZO.n.x̄,
    #approx.nX = ZO.n.Q
    #approx.nX_ = ZO.n.Q_
    #approx.nΘ = ZO.n.Θ
    approx.J = zeros(ZO.n.x,T,ZO.n.Q,T) 
    return approx
end

# Changed c to f

"""
    compute_x_x′_matrices!(FO::FirstOrderApproximation)

Compute c matrices for use in Lemma 2
"""
function compute_f_matrices!(FO::FirstOrderApproximation)
    @unpack ZO,f = FO
    @unpack ẑ,x̄,EΦ′,p, dF = ZO
    N = length(ẑ)
    Ex̄′_z = x̄*EΦ′
    for j in 1:N
        @views f[j] = -inv(dF.x[j] + dF.x′[j]*Ex̄′_z[:,j]*p)
    end
end


"""
    compute_Lemma3!(FO)

Computes the terms from Lemma 3, x_s = dx_t/dX_t+s
"""
# GOOD as before

function compute_Lemma3!(FO::FirstOrderApproximation)
    #compute_f_matrices!(FO) #maybe uncomment and then remove from compute_theta_derivative
    @unpack ZO,f,x,T = FO
    @unpack ẑ,x̄,EΦ,Φ,p,n, dF = ZO
    N = length(ẑ)
    luΦ = lu(Φ) #precompute inverse of basis matrix
    xtemp = zeros(n.x,n.Q,N) #one nx x nQ matrix for each gridpoint
    cFx′ = Vector{Matrix{Float64}}(undef,N)
    
    for i in 1:N
        cFx′[i] = f[i]*dF.x′[i]
        xtemp[:,:,i] .= f[i]*dF.X[i]
    end
    x[1] = xtemp/luΦ

    for s in 2:T
        Ex = x[s-1]*EΦ
        for i in 1:N
            @views xtemp[:,:,i] .= cFx′[i]*Ex[:,:,i]
        end
        x[s] = xtemp/luΦ
    end
end

"""
    compute_Lemma4!(FO)

Computes the terms from Lemma 4, Operators L and terms a_s = M p x_s 
"""
function compute_Lemma4!(FO)
    @unpack ZO,x,T = FO
    @unpack Φz̄,Φ′z̄,p,ω̄,Λ,x̄,n = ZO
    
    #start with computing A 
    z̄_z = ((p*x̄)*Φ′z̄)[:]  #1xIz array should work with broadcasting
    FO.L = L = deepcopy(Λ)
    for j in eachindex(z̄_z)
        for index in nzrange(L,j)
            @inbounds L.nzval[index] *= z̄_z[j]
        end
    end

    #Next compute a objects
    Iz = length(ω̄)
    FO.M = Λ*(Φz̄'.*ω̄)
    
    FO.z = z = zeros(n.ẑ,n.Q,T)
    for s in 1:T
        z[:,:,s] .= (p*x[s])[1,:,:]'
    end
end


"""
    compute_Corollary2!(FO)

Constructs J object from Corollary 1 
"""
function compute_Corollary2!(FO)
    @unpack ZO,T,L,M = FO
    @unpack Φ′z̄,x̄,ω̄ ,n,Φz̄= ZO
    Lt = sparse(L')
    Mt = sparse(M')
    #compute expectations vector
    IL  = zeros(n.z̄,n.x,T)
    IL[:,:,1] = (x̄*Φ′z̄)'
    for t in 2:T
        @views IL[:,:,t] = Lt*IL[:,:,t-1]
    end
    FO.ILM = Mt*IL#MΦ*(Λt*IL)
end 

##
function compute_Proposition1!(FO)
    #compute_Corollary2!(FO)
    @unpack ZO,x,T,J,z,ILM = FO
    @unpack Φz̄,p,ω̄ ,n= ZO

    #Iz = length(ω̄)
    IA = reshape(reshape(ILM,n.ẑ,:)'*reshape(z,n.ẑ,:),n.x,T,n.Q,T)

    IntΦ = Φz̄ * ω̄ #operator to integrate splines over ergodic
    for s in 1:T
        @views J[:,1,:,s] .= x[s]*IntΦ
    end

    #initialize l = 0
    for t in 2:T
        @views J[:,t,:,1] .= IA[:,t-1,:,1]
    end
    for s in 2:T
        for t in 2:T
            @views J[:,t,:,s] .= J[:,t-1,:,s-1] .+ IA[:,t-1,:,s]  
        end
    end
end

"""
    compute_BB!(FO::FirstOrderApproximation)

Computes the BB matrix
"""
function compute_BB!(FO::FirstOrderApproximation)
    @unpack ZO,T,J = FO
    @unpack dG,Q_,Q,n = ZO
    ITT = sparse(I,T,T)
    ITT_ = diagm(-1=>ones(T-1))
    #construct BB matrix
    FO.BB = kron(ITT,dG.x)*reshape(J,n.x*T,:)*kron(ITT,Q) .+ kron(ITT,dG.X) .+ kron(ITT_,dG.X_*Q_);
end



"""
    solve_Xt!(FO::FirstOrderApproximation)

Solves for the path Xt.
"""
function solve_Xt!(FO::FirstOrderApproximation,inputs::Inputs)

    ## ! we assume Δ_0 is the change in the histogram and not the cdf

    @unpack ZO,T,Θ_0,Δ_0,X_0,BB,L = FO
    @unpack x̄,Φz̄,dG,n,Λ = ZO
    @unpack ρ_Θ =inputs

    Δ = zeros(length(Δ_0),T)
    Δ[:,1] .= Δ_0
    for t in 2:T
        @views Δ[:,t] = Λ*Δ[:,t-1] # this is different from the paper because Δ is change in histogram and not CDF
    end

    
    #AA = zeros(n.X,T)
    x̄z̄ = x̄*Φz̄ 
    AA = dG.x*(x̄z̄*Δ) # this is different from the paper because Δ is change in histogram and not CDF
    
    for t in 1:T
        @views AA[:,t] .+= dG.Θ*ρ_Θ^(t-1)*Θ_0
    end
    

    AA[:,1] .+= dG.X_*X_0

    Xt = -BB\AA[:]
    FO.X̄_Z = X̄_Z = reshape(Xt,n.X,T)
end


function compute_x̄_Z_Ω̄_Z!(FO::FirstOrderApproximation)
    @unpack ZO,T,x,X̄_Z,L = FO
    @unpack Q,Φz̄,ω̄,p, n,Λ,Λ_z = ZO
    #Fill objects
    N = length(ZO.ẑ)
    FO.x̄_Z = [zeros(n.x,N) for t in 1:T]
    QX̄_Z = Q*X̄_Z

    for s in 1:T
        x_s = permutedims(x[s],[1,3,2])
        for t in 1:T-(s-1)
            @views FO.x̄_Z[t] .+= x_s * QX̄_Z[:,t+s-1]
        end
    end

    #Next use z̄_Z to construct Ω̄_Z
    Φz̄ω̄ = Φz̄.*ω̄'
    Ω̄_Z = FO.Ω̄_Z = zeros(length(ω̄),T)
    for t in 2:T
        z̄_Zt = p*FO.x̄_Z[t-1]
        @views Ω̄_Z[:,t] = Λ*Ω̄_Z[:,t-1] .+ Λ_z*((z̄_Zt*Φz̄ω̄)[:])# this is different from the paper because Δ is change in histogram and not CDF
    end
end



"""
    compute_Θ_derivatives!(FO)

Computes the derivatives in each Θ direction
"""
function compute_Θ_derivatives!(FO::FirstOrderApproximation,inputs::Inputs)
    @unpack ZO = FO
    @unpack n = ZO

    FO.Ω̄_Θ =  Vector{Matrix{Float64}}(undef,n.Θ)
    FO.x̄_Θ =  Vector{Vector{Matrix{Float64}}}(undef,n.Θ)
    FO.X̄_Θ =  Vector{Matrix{Float64}}(undef,n.Θ)
    FO.X_0 = zeros(n.Q_)
    FO.Δ_0 = zeros(n.z̄)
    compute_f_matrices!(FO)
    compute_Lemma3!(FO)
    compute_Lemma4!(FO)
    compute_Corollary2!(FO)
    compute_Proposition1!(FO)
    compute_BB!(FO)
    
    for i in 1:n.Θ
        FO.Θ_0 = I[1:n.Θ,i] #ith basis vector
        solve_Xt!(FO,inputs)
        compute_x̄_Z_Ω̄_Z!(FO)
        FO.Ω̄_Θ[i] = FO.Ω̄_Z
        FO.x̄_Θ[i] = FO.x̄_Z
        FO.X̄_Θ[i] = FO.X̄_Z
    end
end

## verfied that the code runs until here works before this


### OLD

"""
    compute_E!(FO)

Computes the E objects used to construct G from Corollary 1
"""
function compute_ℰ!(FO)
    @unpack ZO,T,L = FO
    @unpack Φ′z̄,x̄,ω̄ ,n= ZO
    #Iz = length(ω̄)
    #compute expectations vector
    E = zeros(n.x,n.z̄,T)
    E[:,:,1] = x̄*Φ′z̄ #This is the B operator
    for t in 2:T 
       @views E[:,:,t] = E[:,:,t-1]*L
    end
    FO.E = permutedims(E,[1,3,2]) #recast as nx×T×Iz
end


### OLD

"""
    compute_Corollary1!(FO)

Constructs J object from Corollary 1 
"""
function compute_Corollary1!(FO)
    #compute_ℰ!(FO)
    @unpack ZO,x,T,J,a,E = FO
    @unpack Φz̄,p,ω̄ ,n= ZO
    #Iz = length(ω̄)
    ℱ = reshape(reshape(E,:,n.z̄)*reshape(a,n.z̄,:),n.x,T,n.Q,T)

    IntΦ = Φz̄ * ω̄ #operator to integrate splines over ergodic
    for s in 1:T
        @views J[:,1,:,s] .= x[s]*IntΦ
    end

    #initialize l = 0
    for t in 2:T
        @views J[:,t,:,1] .= ℱ[:,t-1,:,1]
    end
    for s in 2:T
        for t in 2:T
            @views J[:,t,:,s] .= J[:,t-1,:,s-1] .+ ℱ[:,t-1,:,s]#ℰ[:,:,t-1]*𝒟[:,:,l]  
        end
    end
end


## GOOD? 





### GOOD

"""
    solve_Xt!(FO::FirstOrderApproximation)

Solves for the path Xt.
"""
function solve_Xt!(FO::FirstOrderApproximation)
    @unpack ZO,T,Θ_0,BB = FO
    @unpack x̄,Φ′z̄,dG,ρ_Θ,n = ZO
    #nX = size(G_X,1) #Note this is the full number of aggregates
    
    AA = zeros(n.x,T)
    for t in 1:T
        @views AA[:,t] .+= G_Θ*ρ_Θ^(t-1)*Θ_0
    end

    Xt = -BB\AA[:]
    FO.X̄_Z = X̄_Z = reshape(Xt,n.X,T)
end








