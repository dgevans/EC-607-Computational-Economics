using Parameters,LinearAlgebra,BasisMatrices,SparseArrays,QuantEcon,Arpack,Roots
using ForwardDiff,Dierckx

Approx1D = Interpoland{Basis{1,Tuple{SplineParams{Array{Float64,1}}}},Array{Float64,1},BasisMatrix{Tensor,SparseArrays.SparseMatrixCSC{Float64,Int64}}}

"""
    Contains the Parameters of the Hugget Model
"""
@with_kw mutable struct AiyagariModelCA
    #SteadyState Parameters
    α::Float64 = 0.3 #curvature of production function
    σ::Float64 = 2. #Risk Aversion
    β::Float64 = 0.99 #Discount Factor
    σ_ϵ::Float64 = 0.13 #standard deviation of income shock
    ρ_ϵ::Float64 = 0.966 #persistence of the productivity shocks
    b̲::Float64  = 0. #Borrowing constraint
    bmax::Float64 = 500.
    Nϵ::Int = 7 #number of gridpoints for the productivity shocks
    Nb::Int = 60 #number of gridpoints for splines
    curv_interp::Float64 = 2.5 #controls spacing for interpolation
    kb::Int = 2 #Spline Order
    curv_hist::Float64 = 2. #controls spacing for histogram
    Ib::Int = 1000 #number of gridpoints for histogram
    R̄::Float64 = 1.01 #Equlibrium gross interest rate
    W̄::Float64 = 1.  #Equilibrium wage.
    T̄::Float64 =0.
    ℐ::Float64=0.
    C̄::Float64 =0.
    Ȳ::Float64 =0.
    Iv::Float64 =0.
    K̄::Float64=0.
    q̄::Float64=0.

    K2Y::Float64 = 2.7*4 #target capital to output ratio
    Θ̄::Float64 = 1. #Level of TFP
    δ::Float64 = 0.1 #depreciation rate
    N̄::Float64 = 1. #Average labor supply
    ϕ::Float64 = 10.
    τ_θ::Float64 = -2. #1 pct increase in output => 2% increase in tax

    #Additional Model Parameters
    ρ_Θ::Matrix{Float64} = 0.8*ones(1,1)
    Σ_Θ::Matrix{Float64} = 0.014^2*ones(1,1)
    μ_Θσσ::Vector{Float64} = zeros(1)
    ρ_ν::Float64 = 0.75 #Persistance of volatility shock
    σ_ν::Float64 = 1.  #Size of volatility shock
    Σ_Θ_ν::Matrix{Float64} = Σ_Θ#time varying risk premium

    #Helpful Objects
    b′grid::Vector{Float64} = zeros(1)#grid vector for endogenous grid
    b_cutoff::Vector{Float64} = zeros(1) #Stores the points at which the borrowing constraint binds
    ĵ::Vector{Int} = Int[]
    j̄::Vector{Int} = Int[]

    #policy functions
    cf::Vector{Spline1D} = Vector{Spline1D}(undef,1)
    λf::Function = (b, s) -> 0.0
    bf::Function = (b, s) -> 0.0
    vf::Function= (b, s) -> 0.0

    #grids
    b̂grid::Vector{Float64} = zeros(1)
    b̄grid::Vector{Float64} = zeros(1)    

    #Objects for first order approximation
    x̄f::Matrix{Interpoland} = Matrix{Interpoland}(undef,1,1)
    x̄::Matrix{Float64} =  zeros(1,1)#policy rules
    X̄::Vector{Float64} = zeros(1) #stedy state aggregates
    V̄::Vector{Interpoland} = Vector{Interpoland}(undef,1)

    p::Matrix{Float64} = [1 0 0]  #projection matrix
    Q_::Matrix{Float64} = [zeros(2,7) I] #projection matrix X->X_ 
    Q::Matrix{Float64} = [I zeros(3,6)] #selector matrix for prices relevant for HH problem
    H::SparseMatrixCSC{Float64,Int64} = spzeros(Ib,Ib)
    Hδ::SparseMatrixCSC{Float64,Int64} = spzeros(Ib,Ib)
    H_z::SparseMatrixCSC{Float64,Int64} = spzeros(Ib,Ib)#TODO: update to n-d z
    H_zz::SparseMatrixCSC{Float64,Int64} = spzeros(Ib,Ib)

    ϵ::Vector{Float64} = ones(1) #vector of productivity levels
    πϵ::Matrix{Float64} = ones(1,1) #transition matrix
    ẑ::Vector{Vector{Float64}} = Vector{Vector{Float64}}(undef,1) #gridpoints for the approximations 
    z̄::Matrix{Float64} = ones(1,1) #gridpoints for the distribution
    ω̄::Vector{Float64} = ones(1) #masses for the stationary distribution
    #Φ matricies to use to evaluate interpolated functions in the approximation
    Φ::SparseMatrixCSC{Float64,Int64} = spzeros(Nb,Nb)
    Φ′::SparseMatrixCSC{Float64,Int64} = spzeros(Nb,Nb)
    EΦ′::SparseMatrixCSC{Float64,Int64} = spzeros(Nb,Nb)
    EΦ::SparseMatrixCSC{Float64,Int64} = spzeros(Nb,Nb)
    Φz̄::SparseMatrixCSC{Float64,Int64} = spzeros(Nb,Nb)
    Φ′z̄::SparseMatrixCSC{Float64,Int64} = spzeros(Nb,Nb)
    #Derivatives of R
    R_x::Matrix{Float64} = [1 0.;
                            0. 0.]
    R_X::Matrix{Float64} = [0. 0.;
                            0. -1.]
    R_X_::Matrix{Float64} = zeros(1,1)
    R_Θ::Matrix{Float64} = [0. 1.]'

    R_xx::Array{Float64,3} = zeros(1,1,1)
    R_xX_::Array{Float64,3} = zeros(1,1,1)
    R_xX::Array{Float64,3} = zeros(1,1,1)
    R_xΘ::Array{Float64,3} = zeros(1,1,1)
    R_X_X_::Array{Float64,3} = zeros(1,1,1)
    R_X_X::Array{Float64,3} = zeros(1,1,1)
    R_X_Θ::Array{Float64,3} = zeros(1,1,1)
    R_XX::Array{Float64,3} = zeros(1,1,1)
    R_XΘ::Array{Float64,3} = zeros(1,1,1)
    R_ΘΘ::Array{Float64,3} = zeros(1,1,1)

end


"""
    checkΦmatrices(AM,V,bf)

A tool for checking if I constructed the basis matrices correctly.
"""
function checkΦmatrices(AM,V,bf)
    @unpack ẑ,z̄,πϵ = AM

    S = length(AM.ϵ)
    #Want to check that we setup the basis matrices correctly
    Vcoefs = hcat([V[s].coefs' for s in 1:S]...)
    #Check Φ
    diffΦ = norm((Vcoefs*AM.Φ)' - [V[Int(z[2])](z[1]) for z in ẑ],Inf)
    @assert diffΦ < 1e-10
    #Check EΦ, EΦ′
    EV = zeros(length(ẑ))
    EV′ = zeros(length(ẑ))
    for i in 1:length(ẑ)
        b,s = ẑ[i]
        b′ = bf(b,Int(s))
        EV[i] = dot(πϵ[Int(s),:],[V[s′](b′) for s′ in 1:S])
        EV′[i] = dot(πϵ[Int(s),:],[V[s′](b′,ones(Int,1,1)) for s′ in 1:S])
    end
    diffEΦ = norm((Vcoefs*AM.EΦ)' - EV,Inf)
    @assert diffEΦ < 1e-10
    diffEΦ′ = norm((Vcoefs*AM.EΦ′)' - EV′,Inf)
    @assert diffEΦ′ < 1e-10

    #check Φz̄
    Vz̄ = zeros(size(z̄,1))
    Vz̄′ = zeros(size(z̄,1))
    for i in 1:length(Vz̄)
        z,s = z̄[i,1],Int(z̄[i,2])
        Vz̄[i] = V[s](z)
        Vz̄′[i] = V[s](z,ones(Int,1,1))
    end
    diffΦz̄ = norm((Vcoefs*AM.Φz̄)' - Vz̄,Inf)
    @assert diffΦz̄ < 1e-10
    diffΦz̄ = norm((Vcoefs*AM.Φ′z̄)' - Vz̄′,Inf)
    @assert diffΦz̄ < 1e-10
end


"""
computeoptimalconsumption(AM::AiyagariModelCA,V)

Computes optimal savings using endogenous grid method.  
"""
function computeoptimalconsumption(AM::AiyagariModelCA,Vcoefs::Vector{Float64})::Vector{Interpoland}
    @unpack σ,β,ϵ,b̲,EΦ′,b′grid,R̄,W̄,b̲ = AM
    S = length(ϵ)
    EV_b = reshape(EΦ′*Vcoefs,:,S) #precomputing expectations
    
    c = (β.*EV_b).^(-1/σ) #consumption today
    b = (b′grid .+ c .- W̄.*ϵ')/R̄  #Implied assets today

    cf = Vector{Interpoland}(undef,S)#implied policy rules for each productivity
    for s in 1:S
        #with some productivities the borrowing constraint does not bind
        if b[1,s] > b̲ #borrowing constraint binds
            AM.b_cutoff[s] = b[1,s]
            #add extra points on the borrowing constraint for interpolation
            b̂ = [b̲;b[:,s]]
            ĉ = [R̄*b̲-b̲ + W̄*ϵ[s];c[:,s]]
            cf[s] = Interpoland(Basis(SplineParams(b̂,0,1)),ĉ)
        else
            AM.b_cutoff[s] = -Inf
            cf[s] = Interpoland(Basis(SplineParams(b[:,s],0,1)),c[:,s])
        end
    end
    return cf
end


"""
computeoptimalconsumption_λ(AM::AiyagariModelCA,λcoefs)

Computes optimal savings using endogenous grid method.  
"""
function computeoptimalconsumption_λ(AM::AiyagariModelCA,λcoefs::Vector{Float64})::Vector{Spline1D}
    @unpack σ,β,ϵ,b̲,EΦ,b′grid,R̄,W̄,b̲ = AM
    S = length(ϵ)
    Eλ′ = reshape(EΦ*λcoefs,:,S) #precomputing expectations
    
    c = (β.*Eλ′).^(-1/σ) #consumption today
    b = (b′grid .+ c .- W̄.*ϵ')/R̄  #Implied assets today

    cf = Vector{Spline1D}(undef,S)#implied policy rules for each productivity
    for s in 1:S
        #with some productivities the borrowing constraint does not bind
        if b[1,s] > b̲ #borrowing constraint binds
            AM.b_cutoff[s] = b[1,s]
            #add extra points on the borrowing constraint for interpolation
            b̂ = [b̲;b[:,s]]
            ĉ = [R̄*b̲-b̲ + W̄*ϵ[s];c[:,s]]
            cf[s] = Spline1D(b̂,ĉ,k=1)
        else
            AM.b_cutoff[s] = -Inf
            cf[s] = Spline1D(b[:,s],c[:,s],k=1)
        end
    end
    return cf
end

"""
    iteratebellman_time!(AM::AiyagariModelCA,Vcoefs)

Updates the coefficients of the value function using time iteration of the bellman equation
"""
function iteratebellman_time!(AM::AiyagariModelCA,V)
    @unpack σ,β,Φ,ϵ,πϵ,R̄,W̄ = AM
    S = length(ϵ)

    Vcoefs = vcat([V[s].coefs for s in 1:S]...)::Vector{Float64}
    bgrid = nodes(V[1].basis)[1]
    Nb = length(bgrid)

    cf = computeoptimalconsumption(AM,Vcoefs) #Compute optimal consumption function
    c = zeros(Nb*S) 
    EΦ = spzeros(Nb*S,Nb*S)
    for s in 1:S
        for s′ in 1:S
            c[(s-1)*Nb+1:s*Nb] = cf[s](bgrid) #compute consumption at gridpoints
            b′ = R̄*bgrid .+ ϵ[s]*W̄ .- c[(s-1)*Nb+1:s*Nb] #asset choice
            EΦ[(s-1)*Nb+1:s*Nb,(s′-1)*Nb+1:s′*Nb] = πϵ[s,s′]*BasisMatrix(V[s].basis,Direct(),b′).vals[1][:]
        end
    end

    res = c.^(1-σ)./(1-σ) .+ β.*EΦ*Vcoefs - Φ*Vcoefs
    Vcoefs′ = Φ\(c.^(1-σ)./(1-σ) .+ β.*EΦ*Vcoefs)
    for s in 1:S
        V[s].coefs .= Vcoefs′[1+(s-1)*Nb:s*Nb]
    end
    return norm(res,Inf)
end


"""
    iteratebellman_time!(AM::AiyagariModelCA,Vcoefs)

Updates the coefficients of the value function using time iteration of the bellman equation
"""
function iteratebellman_eg!(AM::AiyagariModelCA,luΦ,λcoefs)
    @unpack σ,β,Φ,ϵ,πϵ,R̄,W̄ = AM
    S = length(ϵ)

    bgrid = AM.b′grid
    Nb = length(bgrid)

    cf = computeoptimalconsumption_λ(AM,λcoefs) #Compute optimal consumption function
    λ = zeros(Nb*S) 
    for s in 1:S
        λ[(s-1)*Nb+1:s*Nb] = R̄.*cf[s](bgrid).^(-σ) #compute consumption at gridpoints
    end


    λcoefs′ = luΦ\λ
    diff = norm(λcoefs.-λcoefs′,Inf)
    λcoefs .= λcoefs′
    return diff
end


"""
    iteratebellman_newton!(AM::AiyagariModelCA,Vcoefs)

Updates the coefficients of the value function using time iteration of the bellman equation
"""
function iteratebellman_newton!(AM::AiyagariModelCA,V)
    @unpack σ,β,Φ,ϵ,πϵ,R̄,W̄ = AM
    S = length(ϵ)

    Vcoefs = vcat([V[s].coefs for s in 1:S]...)::Vector{Float64}
    bgrid = nodes(V[1].basis)[1]
    Nb = length(bgrid)

    cf = computeoptimalconsumption(AM,Vcoefs) #Compute optimal consumption function
    c = zeros(Nb*S) 
    EΦ = spzeros(Nb*S,Nb*S)
    for s in 1:S
        for s′ in 1:S
            c[(s-1)*Nb+1:s*Nb] = cf[s](bgrid) #compute consumption at gridpoints
            b′ = R̄*bgrid .+ ϵ[s]*W̄ .- c[(s-1)*Nb+1:s*Nb] #asset choice
            EΦ[(s-1)*Nb+1:s*Nb,(s′-1)*Nb+1:s′*Nb] = πϵ[s,s′]*BasisMatrix(V[s].basis,Direct(),b′).vals[1][:]
        end
    end


    Jac = β.*EΦ .- Φ
    res = c.^(1-σ)./(1-σ) .+ Jac*Vcoefs
    Vcoefs′ = Vcoefs - Jac\res #newtons method

    for s in 1:S
        V[s].coefs .= Vcoefs′[1+(s-1)*Nb:s*Nb]
    end
    return norm(res,Inf)  #return norm
end

"""
    solvebellman!(AM::AiyagariModelCA,V)

Solves the bellman equation for given some initial 
value function V.
"""
function solvebellman!(AM::AiyagariModelCA,V,tol=1e-6)
    #increases stability to iterate on the time dimension a few times
    diff = 1.
    while diff > tol
        #then use newtons method
        diff = iteratebellman_newton!(AM,V)
    end
end


"""
    solvebellman!(AM::AiyagariModelCA,V)

Solves the bellman equation for given some initial 
value function V.
"""
function solvebellman_eg!(AM::AiyagariModelCA,λcoefs,tol=1e-8)
    #increases stability to iterate on the time dimension a few times
    diff = 1.
    luΦ = lu(AM.Φ)
    while diff > tol
        #then use newtons method
        diff = iteratebellman_eg!(AM,luΦ,λcoefs)
    end
end


"""
    setupgrids_and_VF!(AM::AiyagariModelCA,bmax,curv=1.7)

Setup grid points for the Hugget Model given parameters
"""
function setupgrids_and_VF!(AM::AiyagariModelCA)
    @unpack b̲,Nb,ρ_ϵ,σ_ϵ,β,σ,Ib,bmax,curv_interp,curv_hist = AM
    S = AM.Nϵ
    xvec = LinRange(0,1,Nb-1).^curv_interp  #The Nb -1 to to adjust for the quadratic splines
    b′grid = b̲ .+ (bmax - b̲).*xvec #nonlinear grides
    
    #Now gridpoints for ϵ
    mc = rouwenhorst(S,ρ_ϵ,σ_ϵ)
    πϵ = AM.πϵ = mc.p
    AM.ϵ = exp.(mc.state_values)
    πstat = real(eigs(πϵ',nev=1)[2])
    πstat ./= sum(πstat)
    AM.ϵ = AM.ϵ./dot(πstat,AM.ϵ)
    AM.N̄ = dot(πstat,AM.ϵ)

    #Gridpointsfor the value function
    bbasis = Basis(SplineParams(b′grid,0,2))
    V = [Interpoland(bbasis,b->((1-β)*b.+1).^(1-σ)./(1-σ)./(1-β)) for s in 1:S] #initialize with value function equal to β*a
    
    bgrid = nodes(bbasis)[1]
    AM.b′grid = bgrid

    #Precompute EΦ′ and EΦ
    AM.EΦ = kron(πϵ,BasisMatrix(bbasis,Direct(),AM.b′grid).vals[1])
    AM.EΦ′ = kron(πϵ,BasisMatrix(bbasis,Direct(),AM.b′grid,[1]).vals[1])
    #Precompute Phi
    AM.Φ = kron(Matrix(I,S,S),BasisMatrix(bbasis,Direct()).vals[1])
    AM.Φ′ = kron(Matrix(I,S,S),BasisMatrix(bbasis,Direct(),bgrid,[1]).vals[1])

    λcoefs = AM.Φ\repeat((1/β).*((1-β)*bgrid.+1).^(-σ),S)
    #Grid for distribution
    xvec = LinRange(0,1,Ib).^curv_hist 
    b̄grid = b̲ .+ (bmax - b̲).*xvec #nonlinear grides
    AM.z̄ = hcat(kron(ones(S),b̄grid),kron(1:S,ones(Ib)))
    AM.ω̄ = ones(Ib*S)/(Ib*S)

    #cutoffs
    AM.b_cutoff = zeros(S) 
    #Do some time iterations so newton's method is stable
    for i in 1:20
        iteratebellman_time!(AM,V)
    end

    return V,λcoefs
end

"""
    find_stationarydistribution!(AM::AiyagariModelCA,V)

Computes the stationary distribution 
"""
function find_stationarydistribution!(AM::AiyagariModelCA,V)
    @unpack ϵ,πϵ,Ib,z̄,R̄,W̄ = AM
    S = length(ϵ)
    Vcoefs = vcat([V[s].coefs for s in 1:S]...)::Vector{Float64}
    cf = computeoptimalconsumption(AM,Vcoefs)
    b̄ = z̄[1:Ib,1] #grids are all the same for all shocks
    c = hcat([cf[s](b̄) for s in 1:S]...) #consumption policy
    b′ = R̄.*b̄ .+ W̄.*ϵ' .- c #create a Ib×S grid for the policy rules
    
    #make sure we don't go beyond bounds.  Shouldn't bind if bmax is correct
    b′ = max.(min.(b′,b̄[end]),b̄[1])
    
    Qb = BasisMatrix(Basis(SplineParams(b̄,0,1)),Direct(),reshape(b′,Ib*S)).vals[1]
    Q = spzeros(Ib*S,Ib*S)
    for s in 1:S
        Q[1+(s-1)*Ib:s*Ib,:] = kron(reshape(πϵ[s,:],1,:),Qb[1+(s-1)*Ib:s*Ib,:]) 
    end
    
    AM.H = Q'
    AM.ω̄ .= real(eigs(AM.H;nev=1)[2])[:]
    AM.ω̄ ./= sum(AM.ω̄) 
end


"""
    find_stationarydistribution!(AM::AiyagariModelCA,V)

Computes the stationary distribution 
"""
function find_stationarydistribution_λ!(AM::AiyagariModelCA,λcoefs)
    @unpack ϵ,πϵ,Ib,z̄,R̄,W̄ = AM
    S = length(ϵ)
    cf = computeoptimalconsumption_λ(AM,λcoefs)::Vector{Spline1D}
    b̄ = z̄[1:Ib,1] #grids are all the same for all shocks
    c = hcat([cf[s](b̄) for s in 1:S]...) #consumption policy
    b′ = R̄.*b̄ .+ W̄.*ϵ' .- c #create a Ib×S grid for the policy rules
    
    #make sure we don't go beyond bounds.  Shouldn't bind if bmax is correct
    b′ = max.(min.(b′,b̄[end]),b̄[1])
    
    Qb = BasisMatrix(Basis(SplineParams(b̄,0,1)),Direct(),reshape(b′,Ib*S)).vals[1]
    Q = spzeros(Ib*S,Ib*S)
    for s in 1:S
        Q[1+(s-1)*Ib:s*Ib,:] = kron(reshape(πϵ[s,:],1,:),Qb[1+(s-1)*Ib:s*Ib,:]) 
    end
    
    AM.H = Q'
    AM.ω̄ .= real(eigs(AM.H;nev=1)[2])[:]
    AM.ω̄ ./= sum(AM.ω̄) 
end


"""
    calibratesteadystate!(AM::AiyagariModelCA)

Solves for the steady state without aggregate shocks
"""
function calibratesteadystate!(AM::AiyagariModelCA)
    V,λcoefs = setupgrids_and_VF!(AM)
    @unpack Θ̄,α,N̄,K2Y,R̄ = AM
    

    #Compute objects from targeted moments
    Y2K = 1/K2Y
    AM.δ = δ = α*Y2K + 1 - R̄ #matching capital to output ratio and interest rate gives depreciation rate
    K2N = (Y2K/Θ̄)^(1/(α-1)) #relationship between capital to output and capital to labor
    K̄ = K2N*N̄
    AM.W̄ = (1-α)*Θ̄*K2N^α

    function βres(β)
        AM.β = β
        solvebellman!(AM,V)
        find_stationarydistribution!(AM,V)

        return dot(AM.ω̄,AM.z̄[:,1]) - K̄
    end
    Q̄ = 1/R̄
    fzero(βres,Q̄^5,Q̄)

    return V
end

"""
    calibratesteadystate!(AM::AiyagariModelCA)

Solves for the steady state without aggregate shocks
"""
function calibratesteadystate_λ!(AM::AiyagariModelCA)
    V,λcoefs = setupgrids_and_VF!(AM)
    @unpack Θ̄,α,N̄,K2Y,R̄ = AM
    

    #Compute objects from targeted moments
    Y2K = 1/K2Y
    AM.δ = δ = α*Y2K + 1 - R̄ #matching capital to output ratio and interest rate gives depreciation rate
    K2N = (Y2K/Θ̄)^(1/(α-1)) #relationship between capital to output and capital to labor
    K̄ = K2N*N̄
    AM.W̄ = (1-α)*Θ̄*K2N^α

    function βres(β)
        AM.β = β
        solvebellman_eg!(AM,λcoefs)
        find_stationarydistribution_λ!(AM,λcoefs)

        return dot(AM.ω̄,AM.z̄[:,1]) - K̄
    end
    Q̄ = 1/R̄
    fzero(βres,Q̄^30,Q̄)
    #also solve for the value function (needed for welfare)
    solvebellman!(AM,V)

    return V,λcoefs
end


"""
    calibratesteadystate!(AM::AiyagariModelCA)

Solves for the steady state without aggregate shocks
"""
function solvesteadystate!(AM::AiyagariModelCA)
    V = setupgrids_and_VF!(AM)
    @unpack α,N̄,K2Y,δ,β,Θ̄ = AM

    function Kres(K)
        K2N = K/N̄
        AM.W̄ = (1-α)*Θ̄*K2N^α
        AM.R̄ = α*Θ̄*K2N^(α-1) + 1 - δ
        solvebellman!(AM,V)
        find_stationarydistribution!(AM,V)

        return dot(AM.ω̄,AM.z̄[:,1]) - K
    end
    K2Nmin = ((1/β-1+δ)/(α*Θ̄))^(1/(α-1))
    Kmin = K2Nmin*N̄
    fzero(Kres,Kmin,Kmin*2)

    return V
end




function save_policy_functions!(AM::AiyagariModelCA)
    @unpack R̄,W̄,ϵ,πϵ,σ,Ib,Nϵ,Nb,curv_interp,curv_hist,b̲,bmax,V̄,z̄= AM #then unpack equilibrium objects

    AM.V̄,_ =  V,λcoefs  = calibratesteadystate_λ!(AM)
    S = length(ϵ)
    Vcoefs = vcat([V[s].coefs for s in 1:S]...)::Vector{Float64}
    cf = computeoptimalconsumption_λ(AM,λcoefs)
    AM.cf=cf
    
    λf(b,s) = cf[s](b).^(-σ)*R̄
    bf(b,s) = R̄*b .+ W̄*ϵ[s] .- cf[s](b) #helper function for debt policy
    vf(b,s) = V̄[s](b)
    AM.λf=λf
    AM.bf=bf
    AM.vf=vf

    xvec = LinRange(0,1,Nb-1).^curv_interp  #The Nb -1 to to adjust for the quadratic splines
    AM.b̂grid = b̲ .+ (bmax - b̲).*xvec #nonlinear grides
    xvec = LinRange(0,1,Ib).^curv_hist 
    AM.b̄grid = b̲ .+ (bmax - b̲).*xvec #nonlinear grides
    
    @unpack ω̄,Nb,ϵ,α,δ,N̄,K2Y,Θ̄ ,Ib,z̄,V̄,R̄,W̄= AM 
    S = length(ϵ)
    Y2K = 1/K2Y
    K2N = (Y2K/Θ̄)^(1/(α-1))
    K̄ = N̄*K2N
    Ȳ = Y2K*K̄
    C̄ = Ȳ - δ*K̄
    v = zeros(Ib*S) 
    #compute aggregate welfare
    for s in 1:S
        for s′ in 1:S
            v[(s-1)*Ib+1:s*Ib] = V̄[s](z̄[1:Ib,1]) #compute consumption at gridpoints
        end
    end
end

function save_agg!(AM::AiyagariModelCA)
    @unpack ω̄,Nb,ϵ,α,δ,N̄,K2Y,Θ̄ ,Ib,z̄,V̄,R̄,W̄, cf, πϵ= AM 
    S = length(ϵ)
    Y2K = 1/K2Y
    K2N = (Y2K/Θ̄)^(1/(α-1))
    K̄ = N̄*K2N
    Ȳ = Y2K*K̄
    C̄ = Ȳ - δ*K̄
    v = zeros(Ib*S) 
    #compute aggregate welfare
    for s in 1:S
        for s′ in 1:S
            v[(s-1)*Ib+1:s*Ib] = V̄[s](z̄[1:Ib,1]) #compute consumption at gridpoints
        end
    end
    Iv = dot(v,ω̄)
    AM.R̄ = R̄
    AM.W̄ = W̄ 
    AM.T̄ = 0.
    AM.ℐ = δ
    AM.C̄ = C̄
    AM.Ȳ = Ȳ
    AM.Iv = Iv
    AM.K̄ = K̄
    AM.q̄ = 1.

end


function save_H!(AM::AiyagariModelCA)
    @unpack R̄,W̄,ϵ,πϵ,σ,Ib,Nϵ,Nb,curv_interp,curv_hist,b̲,bmax,V̄,z̄, cf= AM #then unpack equilibrium objects
    S = length(ϵ)
    bbasis =V̄[1].basis
    ẑ = hcat(kron(ones(S),nodes(bbasis)[1]),kron(1:S,ones(length(bbasis))))
    bgrid = nodes(bbasis)[1]
    N = length(bgrid)

    
    #construct H_z
    b̄ = z̄[1:Ib,1] #grids are all the same for all shocks
    c = hcat([cf[s](b̄) for s in 1:S]...) #consumption policy
    b′ = R̄.*b̄ .+ W̄.*ϵ' .- c #create a Ib×S grid for the policy rules
    
    #make sure we don't go beyond bounds.  Shouldn't bind if bmax is correct
    b′ = max.(min.(b′,b̄[end]),b̄[1])
    b̄basis = Basis(SplineParams(b̄,0,1))
    f(x̂) = BasisMatrix(b̄basis,Direct(),[x̂]).vals[1]

    Qb_b = spzeros(Ib*S,Ib)
    for i in 1:length(b′)
        Qb_b[i,:] = ForwardDiff.derivative(f,b′[i])
    end
    Q_b = spzeros(Ib*S,Ib*S)
    for s in 1:S
        Q_b[1+(s-1)*Ib:s*Ib,:] = kron(reshape(πϵ[s,:],1,:),Qb_b[1+(s-1)*Ib:s*Ib,:]) 
    end
    AM.H_z = Q_b'
    AM.H_zz = 0*AM.H_z


    #Construct bounding js
    â = unique(ẑ[:,1])
    ā = unique(z̄[:,1])
    ĵ = AM.ĵ = Int[]
    j̄ = AM.j̄ = Int[]
    for s in 1:S
        if AM.b_cutoff[s] > -Inf
            push!(ĵ,findlast(â .< AM.b_cutoff[s])+(s-1)*N)
            push!(j̄,findlast(ā .< AM.b_cutoff[s])+(s-1)*Ib)
        end
    end 
    #construct Hδ
    AM.Hδ = AM.H[:,j̄] #transition matrix from kinks


end


##########################################################################################################################################################
## Anmol: BELOW is the part we probably dont need 


"""
The Model

Thing are set up a little weird to try and minimize the number of variables. 
x = [b,λ,b̂]
where λ = c^(-σ)*R and b̂ is the level of assets individuals enter with
X = [R,W,K]

The individual constraints are for states b_,s
F = [R*b_ +W*ϵ[s] - (λ/R)^(-1/σ) - b;
     b_ - b̂;
     β𝐄λ′ - λ/R [OR] b̂-b̲] 
the last constraint depends on whever borrowing constraint binds

The aggregate constraints are
R = [∫b̂(z)dΩ(z) - K;
     1 + αΘK^(α-1)N^(1-α) - δ - R;
     (1-α)Θ*K^α*N^(-α) - W]


The following functions Fcon, Funcon define the model for the constrained and unconstrained agents respectively
R defines the aggregate constraints.  Derivatives are computed via automatic differentiation
"""
function F(M::AiyagariModelCA,s,b_,x,X,x′)
    @unpack b_cutoff,β,σ,ϵ = M
    #unpack variables
    b,λ,v = x
    _,Eλ′,Ev′ = x′
    R,W,T = X

    c = (λ/R)^(-1/σ)


    ret = [R*b_+W*ϵ[s]+T-c-b,
           v - c^(1-σ)/(1-σ) - β*Ev′,
           β*Eλ′-λ/R]
    if b_ < b_cutoff[s]
        ret[3] = M.b̲-b
    end
    return ret
end

function R(M::AiyagariModelCA,Ix,X_,X,Θ)
    @unpack α,δ,N̄,ϕ,τ_θ = M
    qK,_,Iv = Ix
    R,W,T,ℐ,C,Y,V,K,q = X
    K_,q_ = X_
    #now perform operations
    rK = α*Θ[1]*K_^(α-1)*N̄^(1-α)
    ϕK = ℐ +  0.5*ϕ*(ℐ-δ)^2
    τ = τ_θ*(Θ[1]-1)
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

"""
    F_x(AM::AiyagariModelCA,z)

Computes the partial derivative of the F function w.r.t x evaluated at state z.
Will use z to determine if borrowing constraint is binding or not.
"""
function F_x(AM::AiyagariModelCA,i)::Matrix{Float64}
    b_,s = AM.ẑ[i]
    s = Int(s)
    x̄ = AM.x̄*AM.Φ[:,i]
    X̄ = AM.Q*AM.X̄ #only interest rate and wages relevant
    Ex̄′ = AM.x̄*AM.EΦ[:,i]
    return ForwardDiff.jacobian(x->F(AM,s,b_,x,X̄,Ex̄′),x̄)
end


function F_X(AM::AiyagariModelCA,i)::Matrix{Float64}
    b_,s = AM.ẑ[i]
    s = Int(s)
    x̄ = AM.x̄*AM.Φ[:,i]
    X̄ = AM.Q*AM.X̄ #only interest rate and wages relevant
    Ex̄′ = AM.x̄*AM.EΦ[:,i]
    return ForwardDiff.jacobian(X->F(AM,s,b_,x̄,X,Ex̄′),X̄)
end

function F_x′(AM::AiyagariModelCA,i)::Matrix{Float64}
    b_,s = AM.ẑ[i]
    s = Int(s)
    x̄ = AM.x̄*AM.Φ[:,i]
    X̄ = AM.Q*AM.X̄ #only interest rate and wages relevant
    Ex̄′ = AM.x̄*AM.EΦ[:,i]
    return ForwardDiff.jacobian(x′->F(AM,s,b_,x̄,X̄,x′),Ex̄′)
end

function F_z(AM::AiyagariModelCA,i)::Matrix{Float64}
    b_,s = AM.ẑ[i]
    s = Int(s)
    x̄ = AM.x̄*AM.Φ[:,i]
    X̄ = AM.Q*AM.X̄ #only interest rate and wages relevant
    Ex̄′ = AM.x̄*AM.EΦ[:,i]
    return ForwardDiff.derivative(b->F(AM,s,b,x̄,X̄,Ex̄′),b_)
end

function F_zz(AM::AiyagariModelCA,i)::Vector{Float64}
    b_,s = AM.ẑ[i]
    s = Int(s)
    x̄ = AM.x̄*AM.Φ[:,i]
    X̄ = AM.Q*AM.X̄ #only interest rate and wages relevant
    Ex̄′ = AM.x̄*AM.EΦ[:,i]
    return ForwardDiff.derivative(b2->ForwardDiff.derivative(b1->F(AM,s,b1,x̄,X̄,Ex̄′),b2),b_)
end

function F_zx(AM::AiyagariModelCA,i)::Matrix{Float64}
    b_,s = AM.ẑ[i]
    s = Int(s)
    x̄ = AM.x̄*AM.Φ[:,i]
    X̄ = AM.Q*AM.X̄ #only interest rate and wages relevant
    Ex̄′ = AM.x̄*AM.EΦ[:,i]

    F_z = x -> ForwardDiff.derivative(b1->F(AM,s,b1,x,X̄,Ex̄′),b_)
    return ForwardDiff.jacobian(F_z,x̄) 
end

function F_zx′(AM::AiyagariModelCA,i)::Matrix{Float64}
    b_,s = AM.ẑ[i]
    s = Int(s)
    x̄ = AM.x̄*AM.Φ[:,i]
    X̄ = AM.Q*AM.X̄ #only interest rate and wages relevant
    Ex̄′ = AM.x̄*AM.EΦ[:,i]
    F_z = x′ -> ForwardDiff.derivative(b1->F(AM,s,b1,x̄,X̄,x′),b_)
    return ForwardDiff.jacobian(F_z,Ex̄′)
end

function F_xx(AM::AiyagariModelCA,i)::Array{Float64,3}
    b_,s = AM.ẑ[i]
    s = Int(s)
    x̄ = AM.x̄*AM.Φ[:,i]
    X̄ = AM.Q*AM.X̄ #only interest rate and wages relevant
    Ex̄′ = AM.x̄*AM.EΦ[:,i]
    nx = length(x̄)
    F_x = x1 -> ForwardDiff.jacobian(x2->F(AM,s,b_,x2,X̄,Ex̄′),x1)
    return reshape(ForwardDiff.jacobian(F_x,x̄),nx,nx,nx) 
end

function F_xX(AM::AiyagariModelCA,i)::Array{Float64,3}
    b_,s = AM.ẑ[i]
    s = Int(s)
    x̄ = AM.x̄*AM.Φ[:,i]
    X̄ = AM.Q*AM.X̄ #only interest rate and wages relevant
    Ex̄′ = AM.x̄*AM.EΦ[:,i]
    nx = length(x̄)
    nX = length(X̄)
    F_x = X -> ForwardDiff.jacobian(x->F(AM,s,b_,x,X,Ex̄′),x̄)
    return reshape(ForwardDiff.jacobian(F_x,X̄),nx,nx,nX) 
end

function F_XX(AM::AiyagariModelCA,i)::Array{Float64,3}
    b_,s = AM.ẑ[i]
    s = Int(s)
    x̄ = AM.x̄*AM.Φ[:,i]
    X̄ = AM.Q*AM.X̄ #only interest rate and wages relevant
    Ex̄′ = AM.x̄*AM.EΦ[:,i]
    nx = length(x̄)
    nX = length(X̄)

    F_X = X1 -> ForwardDiff.jacobian(X2->F(AM,s,b_,x̄,X2,Ex̄′),X1)
    return reshape(ForwardDiff.jacobian(F_X,X̄),nx,nX,nX) 
end

function F_xx′(AM::AiyagariModelCA,i)::Array{Float64,3}
    b_,s = AM.ẑ[i]
    s = Int(s)
    x̄ = AM.x̄*AM.Φ[:,i]
    X̄ = AM.Q*AM.X̄ #only interest rate and wages relevant
    Ex̄′ = AM.x̄*AM.EΦ[:,i]
    nx = length(x̄)

    F_x′ = x′ -> ForwardDiff.jacobian(x->F(AM,s,b_,x,X̄,x′),x̄)
    return reshape(ForwardDiff.jacobian(F_x′,Ex̄′),nx,nx,nx) 
end

function F_Xx′(AM::AiyagariModelCA,i)::Array{Float64,3}
    b_,s = AM.ẑ[i]
    s = Int(s)
    x̄ = AM.x̄*AM.Φ[:,i]
    X̄ = AM.Q*AM.X̄ #only interest rate and wages relevant
    Ex̄′ = AM.x̄*AM.EΦ[:,i]
    nx = length(x̄)
    nX = length(X̄)

    F_x′ = x′ -> ForwardDiff.jacobian(X->F(AM,s,b_,x̄,X,x′),X̄)
    return reshape(ForwardDiff.jacobian(F_x′,Ex̄′),nx,nX,nx) 
end

function F_x′x′(AM::AiyagariModelCA,i)::Array{Float64,3}
    b_,s = AM.ẑ[i]
    s = Int(s)
    x̄ = AM.x̄*AM.Φ[:,i]
    X̄ = AM.Q*AM.X̄ #only interest rate and wages relevant
    Ex̄′ = AM.x̄*AM.EΦ[:,i]
    nx = length(x̄)

    F_x′ = x1′ -> ForwardDiff.jacobian(x2′->F(AM,s,b_,x̄,X̄,x2′),x1′)
    return reshape(ForwardDiff.jacobian(F_x′,Ex̄′),nx,nx,nx) 
end


function computeRmatrices!(AM::AiyagariModelCA)
    #construct R derivatives
    @unpack α,δ,N̄,K2Y,Θ̄,x̄,ω̄,Φz̄ = AM 
    Y2K = 1/K2Y
    K2N = (Y2K/Θ̄)^(1/(α-1))
    K̄ = N̄*K2N
    Ȳ = Y2K*K̄
    C̄ = Ȳ - δ*K̄
    Ix̄ = x̄*Φz̄*ω̄
    #R,W,T,ℐ,C,q,V,K,q
    AM.X̄ = X̄ = [AM.R̄,AM.W̄,0.,AM.δ,C̄,Ȳ,Ix̄[end],K̄,1.]
    X̄_ = AM.Q_*X̄
    
    AM.R_x = ForwardDiff.jacobian(x->R(AM,x,X̄_,X̄,[Θ̄]),Ix̄) 
    AM.R_X_ = ForwardDiff.jacobian(X_->R(AM,Ix̄,X_,X̄,[Θ̄]),X̄_) 
    AM.R_X = ForwardDiff.jacobian(X->R(AM,Ix̄,X̄_,X,[Θ̄]),X̄)
    AM.R_Θ = ForwardDiff.jacobian(Θ->R(AM,Ix̄,X̄_,X̄,Θ),[Θ̄])

    nx = length(Ix̄)
    nX = length(X̄)
    nX_ = length(X̄_)
    nΘ = 1
    AM.R_xx = reshape(ForwardDiff.jacobian(x2->ForwardDiff.jacobian(x1->R(AM,x1,X̄_,X̄,[Θ̄]),x2),Ix̄),nX,nx,nx)
    AM.R_xX_ = reshape(ForwardDiff.jacobian(X_->ForwardDiff.jacobian(x->R(AM,x,X_,X̄,[Θ̄]),Ix̄),X̄_),nX,nx,nX_)
    AM.R_xX = reshape(ForwardDiff.jacobian(X->ForwardDiff.jacobian(x->R(AM,x,X̄_,X,[Θ̄]),Ix̄),X̄),nX,nx,nX)
    AM.R_xΘ = reshape(ForwardDiff.jacobian(Θ->ForwardDiff.jacobian(x->R(AM,x,X̄_,X̄,Θ),Ix̄),[Θ̄]),nX,nx,nΘ)
    AM.R_X_X_ = reshape(ForwardDiff.jacobian(X2_->ForwardDiff.jacobian(X1_->R(AM,Ix̄,X1_,X̄,[Θ̄]),X2_),X̄_),nX,nX_,nX_)
    AM.R_X_X = reshape(ForwardDiff.jacobian(X->ForwardDiff.jacobian(X_->R(AM,Ix̄,X_,X,Θ̄),X̄_),X̄),nX,nX_,nX)
    AM.R_X_Θ = reshape(ForwardDiff.jacobian(Θ->ForwardDiff.jacobian(X_->R(AM,Ix̄,X_,X̄,Θ),X̄_),[Θ̄]),nX,nX_,nΘ)
    AM.R_XX = reshape(ForwardDiff.jacobian(X2->ForwardDiff.jacobian(X1->R(AM,Ix̄,X̄_,X1,[Θ̄]),X2),X̄),nX,nX,nX)
    AM.R_XΘ = reshape(ForwardDiff.jacobian(Θ->ForwardDiff.jacobian(X->R(AM,Ix̄,X̄_,X,Θ),X̄),[Θ̄]),nX,nX,nΘ)
    AM.R_ΘΘ = reshape(ForwardDiff.jacobian(Θ2->ForwardDiff.jacobian(Θ1->R(AM,Ix̄,X̄_,X̄,Θ1),Θ2),[Θ̄]),nX,nΘ,nΘ)
    
end



"""
    setup_approximation!(AM)

Solves for the steady state and sets up all variables necessary
for the approximation.
"""
function setup_approximation!(AM::AiyagariModelCA)
    #first solve for the steady state
    AM.V̄,_ =  V,λcoefs  = calibratesteadystate_λ!(AM)
    #AM.V̄=  V  = calibratesteadystate!(AM)
    
    @unpack R̄,W̄,ϵ,πϵ,σ,Ib,z̄ = AM #then unpack equilibrium objects
    S = length(ϵ)
    Vcoefs = vcat([V[s].coefs for s in 1:S]...)::Vector{Float64}
    #cf = computeoptimalconsumption(AM,Vcoefs)
    cf = computeoptimalconsumption_λ(AM,λcoefs)
    λf(b,s) = cf[s](b).^(-σ)*R̄
    bf(b,s) = R̄*b .+ W̄*ϵ[s] .- cf[s](b) #helper function for debt policy
    b_f(b,s) = b
    vf(b,s) = V[s](b)
   

    #now interpolate using quadratic splines so we can take derivatives
    
    bbasis = V[1].basis
    ẑ = hcat(kron(ones(S),nodes(bbasis)[1]),kron(1:S,ones(length(bbasis))))

    AM.cf=cf
    AM.ẑ = [ẑ[i,:] for i in 1:size(ẑ,1)]
    AM.x̄f = Matrix{Interpoland}(undef,3,S)
    AM.x̄f[1,:] .= [Interpoland(bbasis,b->bf(b,s)) for s in 1:S]
    AM.x̄f[2,:] .= [Interpoland(bbasis,b->λf(b,s)) for s in 1:S]
    AM.x̄f[3,:] .= [Interpoland(bbasis,b->vf(b,s)) for s in 1:S]

    AM.x̄ = [hcat([AM.x̄f[1,s].coefs' for s in 1:S]...); #line up coefficients correctly
            hcat([AM.x̄f[2,s].coefs' for s in 1:S]...);
            hcat([AM.x̄f[3,s].coefs' for s in 1:S]...)] 
    
    
    #Now compute Φ BasisMatrices
    bgrid = nodes(bbasis)[1]
    N = length(bgrid)
    #
    #AM.Φ should allready be correct
    AM.Φ = kron(Matrix(I,S,S),BasisMatrix(bbasis,Direct()).vals[1])'
    AM.Φ′ = kron(Matrix(I,S,S),BasisMatrix(bbasis,Direct(),bgrid,[1]).vals[1])'
    EΦ = spzeros(N*S,N*S)
    EΦ′ = spzeros(N*S,N*S)
    for s in 1:S
        for s′ in 1:S
            b′ = R̄*bgrid .+ ϵ[s]*W̄ .- cf[s](bgrid) #asset choice
            EΦ[(s-1)*N+1:s*N,(s′-1)*N+1:s′*N] = πϵ[s,s′]*BasisMatrix(bbasis,Direct(),b′).vals[1]
            EΦ′[(s-1)*N+1:s*N,(s′-1)*N+1:s′*N] = πϵ[s,s′]*BasisMatrix(bbasis,Direct(),b′,[1]).vals[1]
        end
    end
    #Recall our First order code assumes these are transposed
    AM.EΦ = EΦ'
    AM.EΦ′ = (EΦ′)'  

    AM.Φz̄ = kron(Matrix(I,S,S),BasisMatrix(bbasis,Direct(),unique(AM.z̄[:,1])).vals[1])' #note transponse again
    AM.Φ′z̄ =kron(Matrix(I,S,S),BasisMatrix(bbasis,Direct(),unique(AM.z̄[:,1]),[1]).vals[1])' #note transponse again
    #checkΦmatrices(AM,V,bf) #Double check that Φ matrices where constructed correctly
    
    computeRmatrices!(AM)

    #construct H_z
    b̄ = AM.z̄[1:Ib,1] #grids are all the same for all shocks
    c = hcat([cf[s](b̄) for s in 1:S]...) #consumption policy
    b′ = R̄.*b̄ .+ W̄.*ϵ' .- c #create a Ib×S grid for the policy rules
    
    #make sure we don't go beyond bounds.  Shouldn't bind if bmax is correct
    b′ = max.(min.(b′,b̄[end]),b̄[1])
    b̄basis = Basis(SplineParams(b̄,0,1))
    f(x̂) = BasisMatrix(b̄basis,Direct(),[x̂]).vals[1]

    Qb_b = spzeros(Ib*S,Ib)
    for i in 1:length(b′)
        Qb_b[i,:] = ForwardDiff.derivative(f,b′[i])
    end
    Q_b = spzeros(Ib*S,Ib*S)
    for s in 1:S
        Q_b[1+(s-1)*Ib:s*Ib,:] = kron(reshape(πϵ[s,:],1,:),Qb_b[1+(s-1)*Ib:s*Ib,:]) 
    end
    AM.H_z = Q_b'
    AM.H_zz = 0*AM.H_z


    #Construct bounding js
    â = unique(ẑ[:,1])
    ā = unique(z̄[:,1])
    ĵ = AM.ĵ = Int[]
    j̄ = AM.j̄ = Int[]
    for s in 1:S
        if AM.b_cutoff[s] > -Inf
            push!(ĵ,findlast(â .< AM.b_cutoff[s])+(s-1)*N)
            push!(j̄,findlast(ā .< AM.b_cutoff[s])+(s-1)*Ib)
        end
    end 
    #construct Hδ
    AM.Hδ = AM.H[:,j̄] #transition matrix from kinks
end

