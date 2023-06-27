using Parameters,BasisMatrices,SparseArrays,SuiteSparse,TensorOperations


"""
Nums stores sizes of various objects

nx  number of individual policy function
nX  number of aggregate policy functions
nQ  number of aggregate variables that appear in individual problem
nQ_ number of aggregate predetermined variables 
nz̄  number of points on the for storing the histogram
nẑ  number of splines
nθ  number of points on shock grid
  
"""


@with_kw mutable struct Nums
    x::Int64   = 0
    X::Int64   = 0    
    Q::Int64   = 0
    Q_::Int64  = 0
    z̄::Int64   = 0
    ẑ::Int64   = 0
    θ::Int64   = 0 
    Θ::Int64   = 0
    z::Int64   = 0
end



"""
ModelParams is a stuct that stores parameters that show up in F and R directly

"""

@with_kw mutable struct ModelParams
    b_cutoff::Vector{Float64} = zeros(1)
    b̲::Float64 = 0.
    β::Float64 = 0.
    σ::Float64 = 0.
    ϵ::Vector{Float64} = zeros(1)
    α::Float64 = 0.
    δ::Float64 = 0.
    N̄::Float64 = 0.
    ϕ::Float64 = 0.
    τ_θ::Float64 = 0.

end


"""
Imputs is a stuct that contains the user inputs 
"""

@with_kw mutable struct Inputs
    xf::Vector{Function}    = [(b,s)->0] 
    ix::Dict{Symbol, Int64} = Dict()
    iz::Int64               = 0
    ẑgrid::Vector{Float64}  = zeros(1)
    z̄grid::Vector{Float64}  = zeros(1)
    X̄::Vector{Float64}      = zeros(1)
    iX::Dict{Symbol, Int64} = Dict()
    iQ::Vector{Int64}       = ones(1)
    iQ_::Vector{Int64}      = ones(1)
    ω̄ ::Vector{Float64}     = zeros(1)
    πθ::Matrix{Float64}     = zeros(1,1)
    Θ̄::Vector = ones(1)
    ρ_Θ::Matrix{Float64}    = ones(1,1)
    Σ_Θ::Matrix{Float64}    = ones(1,1)
    F::Function             = (para,z_,θ,x,X,x′)->zeros(1)
    G::Function             = (para,Ix,X_,X,Θ)->zeros(1)
    
 end



"""
DerivativesF stores derivatives of F 
"""


@with_kw mutable struct DerivativesF
    x::Vector{Matrix{Float64}}      =  [zeros(1,1)]
    X::Vector{Matrix{Float64}}      =  [zeros(1,1)]
    x′::Vector{Matrix{Float64}}     =  [zeros(1,1)]
    z::Vector{Matrix{Float64}}      =  [zeros(1,1)]
    zz::Vector{Vector{Float64}}     =  [zeros(1)]
    zx::Vector{Matrix{Float64}}     =  [zeros(1,1)]
    zx′::Vector{Matrix{Float64}}    =  [zeros(1,1)]
    xx::Vector{Array{Float64,3}}    =  [zeros(1,1,1)]
    xX::Vector{Array{Float64,3}}    =  [zeros(1,1,1)]
    XX::Vector{Array{Float64,3}}    =  [zeros(1,1,1)]
    xx′::Vector{Array{Float64,3}}   =  [zeros(1,1,1)]
    Xx′::Vector{Array{Float64,3}}   =  [zeros(1,1,1)]
    x′x′::Vector{Array{Float64,3}}  =  [zeros(1,1,1)]
end

"""
DerivativesG stores derivatives of  G
"""

@with_kw mutable struct DerivativesG

    x::Matrix{Float64}      = zeros(1,1)
    X::Matrix{Float64}      = zeros(1,1) 
    X_::Matrix{Float64}     = zeros(1,1)
    Θ::Matrix{Float64}      = zeros(1,1)
    xx::Array{Float64,3}    = zeros(1,1,1)
    xX_::Array{Float64,3}   = zeros(1,1,1)
    xX::Array{Float64,3}    = zeros(1,1,1)
    xΘ::Array{Float64,3}    = zeros(1,1,1)
    X_X_::Array{Float64,3}  = zeros(1,1,1)
    X_X::Array{Float64,3}   = zeros(1,1,1)
    X_Θ::Array{Float64,3}   = zeros(1,1,1)
    XX::Array{Float64,3}    = zeros(1,1,1)
    XΘ::Array{Float64,3}    = zeros(1,1,1)
    ΘΘ::Array{Float64,3}    = zeros(1,1,1)
    
    
end



"""
The Zeroth order class that contains the objects that we need from the zeroth order 
"""


@with_kw mutable struct ZerothOrderApproximation
    # Nums
    n::Nums=Nums()
    # grids
    ẑ::Vector{Vector{Float64}} = Vector{Vector{Float64}}(undef,1) #gridpoints for the approximations 
    z̄::Matrix{Float64} = ones(1,1) #gridpoints for the distribution
    
    #policy functions
    x̄::Matrix{Float64} =  zeros(1,1)#policy rules

    #aggregates
    X̄::Vector{Float64} = zeros(1) #stedy state aggregates
    
    
    #masses for the stationary distribution
    ω̄::Vector{Float64} = ones(1) 

    
    #basis and transition matricies 
    Φ::SparseMatrixCSC{Float64,Int64}   = spzeros(Nb,Nb)
    Φ′::SparseMatrixCSC{Float64,Int64}  = spzeros(Nb,Nb)
    EΦ′::SparseMatrixCSC{Float64,Int64} = spzeros(Nb,Nb)
    EΦ::SparseMatrixCSC{Float64,Int64}  = spzeros(Nb,Nb)
    Φz̄::SparseMatrixCSC{Float64,Int64}  = spzeros(Nb,Nb)
    Φ′z̄::SparseMatrixCSC{Float64,Int64} = spzeros(Nb,Nb)
    Λ::SparseMatrixCSC{Float64,Int64}   = spzeros(Ib,Ib)
    Λ_z::SparseMatrixCSC{Float64,Int64} = spzeros(Ib,Ib)
    Λ_zz::SparseMatrixCSC{Float64,Int64} = spzeros(Ib,Ib)


    
    
    
    #Objects for first order approximation
    p::Matrix{Float64} = zeros(1)'  #projection matrix
    Q_::Matrix{Float64} = zeros(1,1) #projection matrix X->X_ 
    Q::Matrix{Float64} = zeros(1,1) #selector matrix for prices relevant for HH problem
    
    # F and R 

    dF::DerivativesF = DerivativesF() 
    dG::DerivativesG = DerivativesG()

end


"""
Helpful functions
"""


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

function create_array_with_one(n::Int, position::Int)
    arr = zeros(n)  # Create an array of zeros of length n
    arr[position] = 1  # Set the specified position to 1
    return arr
end


function construct_selector_matrix(n::Int64, indices::Vector)
    m = length(indices)
    sel_matrix = sparse(1:m, indices, 1, m, n)
    return sel_matrix
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

function construct_xfs(AM::AiyagariModelCA)
    @unpack λf,bf,vf,b̂grid,b̄grid= AM #then unpack equilibrium objects
    ẑgrid =b̂grid
    z̄grid = b̄grid
    xf=[bf,λf,vf]
    ix=Dict(:b=>1,:λ=>2,:v=>3)
    iz=ix[:b]
    return xf,ix,iz,ẑgrid,z̄grid

end



function construct_zbasis(zgrid::Vector)::Basis{1, Tuple{SplineParams{Vector{Float64}}}}
    zbasis = Basis(SplineParams(zgrid,0,2))
    return zbasis
end
    

function construct_zs(z̄grid::Vector{Float64},ẑgrid::Vector{Float64},n::Nums)
    zbasis=construct_zbasis(ẑgrid)
    ẑtemp = hcat(kron(ones(n.θ),nodes(zbasis)[1]),kron(1:n.θ,ones(length(zbasis))))
    ẑ = [ẑtemp[i,:] for i in 1:size(ẑtemp,1)]
    z̄ = hcat(kron(ones(n.θ),z̄grid),kron(1:n.θ,ones(n.z̄)))
    return ẑ,z̄
end


function construct_x̄s(zbasis::Basis{1, Tuple{SplineParams{Vector{Float64}}}},xf::Vector{Function},n::Nums,inputs::Inputs)
    @unpack iz = inputs
    x̄f = Matrix{Interpoland}(undef,n.x,n.θ)
    x̄=zeros(n.x,n.ẑ)
    for i in 1:n.x
        x̄f[i,:] .= [Interpoland(zbasis,z->xf[i](z,s)) for s in 1:n.θ]
        x̄[i,:]  = hcat([x̄f[i,s].coefs' for s in 1:n.θ]...)
    end
    return x̄
end

function construct_Φs(zbasis::Basis{1, Tuple{SplineParams{Vector{Float64}}}},z̄::Matrix{Float64},xf::Vector{Function},πθ::Matrix{Float64},n::Nums,inputs::Inputs)
    @unpack iz = inputs
    zgrid = nodes(zbasis)[1]
    uniquez̄  = unique(z̄[:,1])


    N = length(zgrid)

    Φ = kron(Matrix(I,n.θ,n.θ),BasisMatrix(zbasis,Direct()).vals[1])'
    Φ′ = kron(Matrix(I,n.θ,n.θ),BasisMatrix(zbasis,Direct(),zgrid,[1]).vals[1])'
    EΦ = spzeros(N*n.θ,N*n.θ)
    EΦ′ = spzeros(N*n.θ,N*n.θ)
    for s in 1:n.θ
        for s′ in 1:n.θ
            #b′ = R̄*bgrid .+ ϵ[s]*W̄ .- cf[s](bgrid) #asset choice
            z′ = xf[iz](zgrid,s)
            EΦ[(s-1)*N+1:s*N,(s′-1)*N+1:s′*N] = πθ[s,s′]*BasisMatrix(zbasis,Direct(),z′).vals[1]
            EΦ′[(s-1)*N+1:s*N,(s′-1)*N+1:s′*N] = πθ[s,s′]*BasisMatrix(zbasis,Direct(),z′,[1]).vals[1]
        end
    end
    #Recall our First order code assumes these are transposed
    EΦ = EΦ'
    EΦ′ = (EΦ′)'  

    Φz̄ = kron(Matrix(I,n.θ,n.θ),BasisMatrix(zbasis,Direct(),uniquez̄[:,1]).vals[1])' #note transponse again
    Φ′z̄ =kron(Matrix(I,n.θ,n.θ),BasisMatrix(zbasis,Direct(),uniquez̄[:,1],[1]).vals[1])' #note transponse again
    return Φ,Φ′,EΦ,EΦ′,Φz̄,Φ′z̄
end 



function ZerothOrderApproximation(inputs::Inputs)
    @unpack xf,ix,iz,ẑgrid,z̄grid = inputs
    @unpack X̄, iX, iQ, iQ_ = inputs
    @unpack ω̄ , πθ, Θ̄ = inputs    
    ZO =    ZerothOrderApproximation()
    
    n=Nums(θ = size(πθ)[1], z̄ = length(z̄grid) , ẑ = (length(ẑgrid)+1)*size(πθ)[1], x=length(xf), X=length(X̄), Q=length(iQ),  Q_=length(iQ_), Θ=length(Θ̄) )
    ẑ,z̄ = construct_zs(z̄grid,ẑgrid,n)
    n.z̄=length(z̄[:,1])
    zbasis=construct_zbasis(ẑgrid)
    x̄ = construct_x̄s(zbasis,xf,n,inputs)
    Φ,Φ′,EΦ,EΦ′,Φz̄,Φ′z̄ = construct_Φs(zbasis,z̄,xf,πθ,n,inputs)
    
    ZO.n=n
    ZO.ẑ = ẑ
    ZO.z̄ = z̄
    ZO.x̄ = x̄
    ZO.X̄ = X̄
    ZO.ω̄ = AM.ω̄ 


    ZO.Φ = Φ
    ZO.Φ′ = Φ′
    ZO.EΦ′ = EΦ′ 
    ZO.EΦ = EΦ
    ZO.Φz̄ = Φz̄
    ZO.Φ′z̄ = Φ′z̄
    ZO.Λ = AM.H
    ZO.Λ_z = AM.H_z

    

    ZO.p=create_array_with_one(n.x,iz)'
    ZO.Q_ = construct_selector_matrix(n.X,iQ_)
    ZO.Q = construct_selector_matrix(n.X,iQ)
        
    return ZO    
end



function computeDerivativesF!(ZO::ZerothOrderApproximation,inputs::Inputs)
    @unpack n,ẑ,Φ,X̄,x̄,EΦ,Q = ZO
    @unpack F = inputs
    dF=DerivativesF()
    dF.x=[zeros(n.x,n.x) for _ in 1:n.ẑ]
    dF.x′=[zeros(n.x,n.x) for _ in 1:n.ẑ]
    dF.X=[zeros(n.x,n.Q) for _ in 1:n.ẑ] # check
    dF.zz = [zeros(n.x) for _ in 1:n.ẑ]
    dF.zx= [zeros(n.x,n.x) for _ in 1:n.ẑ]
    dF.zx′= [zeros(n.x,n.x) for _ in 1:n.ẑ]
    dF.xx = [zeros(n.x,n.x,n.x) for _ in 1:n.ẑ]
    dF.xX = [zeros(n.x,n.x,n.Q) for _ in 1:n.ẑ]
    dF.XX = [zeros(n.x,n.Q,n.Q) for _ in 1:n.ẑ]
    dF.xx′= [zeros(n.x,n.x,n.x) for _ in 1:n.ẑ]
    dF.Xx′= [zeros(n.x,n.Q,n.x) for _ in 1:n.ẑ]
    dF.x′x′=[zeros(n.x,n.x,n.x) for _ in 1:n.ẑ]


    for j in 1:n.ẑ
        b_,θ = ẑ[j]
        θ = Int(θ)
        argx̄ = x̄*Φ[:,j]
        argX̄= Q*X̄ #only interest rate and wages relevant
        argEx̄′ = x̄*EΦ[:,j]
        
        # first order
        @views dF.x[j]      = ForwardDiff.jacobian(x->F(θ,b_,x,argX̄,argEx̄′),argx̄)
        @views dF.x′[j]     = ForwardDiff.jacobian(x′->F(θ,b_,argx̄,argX̄,x′),argEx̄′)
        @views dF.X[j]      = ForwardDiff.jacobian(X->F(θ,b_,argx̄,X,argEx̄′),argX̄)
        
        # second order
        @views dF.zz[j]     = ForwardDiff.derivative(b2->ForwardDiff.derivative(b1->F(θ,b1,argx̄,argX̄,argEx̄′),b2),b_)
        @views dF.zx[j]    = ForwardDiff.jacobian(x -> ForwardDiff.derivative(b1->F(θ,b1,x,argX̄,argEx̄′),b_),argx̄)
        @views dF.zx′[j]    = ForwardDiff.jacobian(x′ -> ForwardDiff.derivative(b1->F(θ,b1,argx̄,argX̄,x′),b_),argEx̄′)
        @views dF.xx[j]     = reshape(ForwardDiff.jacobian(x1 -> ForwardDiff.jacobian(x2->F(θ,b_,x2,argX̄,argEx̄′),x1),argx̄),n.x,n.x,n.x)
        @views dF.xX[j]     = reshape(ForwardDiff.jacobian(X -> ForwardDiff.jacobian(x->F(θ,b_,x,X,argEx̄′),argx̄),argX̄),n.x,n.x,n.Q)
        @views dF.XX[j]     = reshape(ForwardDiff.jacobian(X1 -> ForwardDiff.jacobian(X2->F(θ,b_,argx̄,X2,argEx̄′),X1),argX̄),n.x,n.Q,n.Q)
        @views dF.xx′[j]    = reshape(ForwardDiff.jacobian(x′ -> ForwardDiff.jacobian(x->F(θ,b_,x,argX̄,x′),argx̄),argEx̄′),n.x,n.x,n.x) 
        @views dF.Xx′[j]    = reshape(ForwardDiff.jacobian(x′ -> ForwardDiff.jacobian(X->F(θ,b_,argx̄,X,x′),argX̄),argEx̄′),n.x,n.Q,n.x)  
        @views dF.x′x′[j]   = reshape(ForwardDiff.jacobian(x1′ -> ForwardDiff.jacobian(x2′->F(θ,b_,argx̄,argX̄,x2′),x1′),argEx̄′),n.x,n.x,n.x)
    end
   
    ZO.dF=dF;
end

function computeDerivativesG!(ZO::ZerothOrderApproximation,inputs::Inputs)
    #construct F derivatives
    @unpack n, X̄, x̄, Φz̄,  Q_ = ZO
    @unpack ω̄, Θ̄, G = inputs

    dG = DerivativesG()
    argΘ̄=Θ̄[1]

    X̄_ = Q_*X̄
    Ix̄ = x̄*Φz̄*ω̄

    #first order
    dG.x = ForwardDiff.jacobian(x->G(x,X̄_,X̄,[argΘ̄]),Ix̄) 
    dG.X_ = ForwardDiff.jacobian(X_->G(Ix̄,X_,X̄,[argΘ̄]),X̄_) 
    dG.X = ForwardDiff.jacobian(X->G(Ix̄,X̄_,X,[argΘ̄]),X̄)
    dG.Θ = ForwardDiff.jacobian(Θ->G(Ix̄,X̄_,X̄,Θ),[argΘ̄])

    #second order
    dG.xx   = reshape(ForwardDiff.jacobian(x2->ForwardDiff.jacobian(x1->G(x1,X̄_,X̄,[argΘ̄]),x2),Ix̄),n.X,n.x,n.x)
    dG.xX_  = reshape(ForwardDiff.jacobian(X_->ForwardDiff.jacobian(x->G(x,X_,X̄,[argΘ̄]),Ix̄),X̄_),n.X,n.x,n.Q_)
    dG.xX   = reshape(ForwardDiff.jacobian(X->ForwardDiff.jacobian(x->G(x,X̄_,X,[argΘ̄]),Ix̄),X̄),n.X,n.x,n.X)
    dG.xΘ   = reshape(ForwardDiff.jacobian(Θ->ForwardDiff.jacobian(x->G(x,X̄_,X̄,Θ),Ix̄),[argΘ̄]),n.X,n.x,n.Θ)
    dG.X_X_ = reshape(ForwardDiff.jacobian(X2_->ForwardDiff.jacobian(X1_->G(Ix̄,X1_,X̄,[argΘ̄]),X2_),X̄_),n.X,n.Q_,n.Q_)
    dG.X_X  = reshape(ForwardDiff.jacobian(X->ForwardDiff.jacobian(X_->G(Ix̄,X_,X,Θ̄),X̄_),X̄),n.X,n.Q_,n.X)
    dG.X_Θ  = reshape(ForwardDiff.jacobian(Θ->ForwardDiff.jacobian(X_->G(Ix̄,X_,X̄,Θ),X̄_),[argΘ̄]),n.X,n.Q_,n.Θ)
    dG.XX   = reshape(ForwardDiff.jacobian(X2->ForwardDiff.jacobian(X1->G(Ix̄,X̄_,X1,[argΘ̄]),X2),X̄),n.X,n.X,n.X)
    dG.XΘ   = reshape(ForwardDiff.jacobian(Θ->ForwardDiff.jacobian(X->G(Ix̄,X̄_,X,Θ),X̄),[argΘ̄]),n.X,n.X,n.Θ)
    dG.ΘΘ   = reshape(ForwardDiff.jacobian(Θ2->ForwardDiff.jacobian(Θ1->G(Ix̄,X̄_,X̄,Θ1),Θ2),[argΘ̄]),n.X,n.Θ,n.Θ)

    ZO.dG=dG;
    
end
