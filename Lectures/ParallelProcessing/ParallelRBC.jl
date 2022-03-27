using BasisMatrices,Optim,LinearAlgebra,MPI,Parameters
MPI.Init() #Need this if running on its own


@with_kw mutable struct NCParameters
    A::Float64 = 1.
    α::Float64 = 0.3
    β::Float64 = 0.96
    kgrid::Vector{Float64} = LinRange(0.05,0.5,20)
    spline_order::Int = 3
end


"""
    optimalpolicy(para::NCParameters,Vprime,k)

Computes optimal policy using continuation value function V and current capital
level k given parameters in para.
"""
function optimalpolicy(para::NCParameters,Vprime,k)
    @unpack A,α,β,kgrid = para
    k_bounds = [kgrid[1],kgrid[end]]
    f_objective(kprime) = -( log(A*k^α-kprime)+β*Vprime(kprime) ) #stores objective as function
    k_max = min(A*k^α-.001,k_bounds[2]) #Can't have negative consumptions
    result = optimize(f_objective,k_bounds[1],k_max)
    return (kprime = result.minimizer,V=-result.minimum) #using named tuples 
end

"""
bellmanmap_mpi(Vprime,para::NCParameters)

Apply the bellman map given continuation value function Vprime
"""
function bellmanmap_mpi(para::NCParameters,Vprime::Interpoland)
    comm = MPI.COMM_WORLD
    r,s = MPI.Comm_rank(comm),MPI.Comm_size(comm) #get size of rank of mpi
    kbasis = Vprime.basis
    #sometimes it's helpful to tell julia what type a variable is
    knodes = nodes(kbasis)[1]::Vector{Float64}
    Nk = Int(length(knodes)/s) #note will spit out error if not equally divisible
    mynodes = knodes[1+r*Nk:(r+1)*Nk]#select a range of size Nk from all the nodes

    my_V = map(k->optimalpolicy(para,Vprime,k).V,mynodes)#only compute value function on my nodes

    #Gather at the end
    V = MPI.Allgather(my_V,comm)#will be a vector for each process 
    return Interpoland(kbasis,V) #each process constructs own value function
end

"""
    solvebellman_mpi(para::NCParameters,V0::Interpoland)

Solves the bellman equation for a given V0
"""
function solvebellman_mpi(para::NCParameters,V0::Interpoland)
    comm = MPI.COMM_WORLD
    r,s = MPI.Comm_rank(comm),MPI.Comm_size(comm) #get size of rank of mpi
    diff = 1
    #Iterate of Bellman Map until difference in coefficients goes to zero
    while diff > 1e-6
        V = bellmanmap_mpi(para,V0)
        diff = norm(V.coefs-V0.coefs,Inf)
        V0 = V
    end
    kbasis = V0.basis
    knodes = nodes(kbasis)[1]::Vector{Float64}
    Nk = Int(length(knodes)/s) #note will spit out error if not equally divisible
    mynodes = knodes[1+r*Nk:(r+1)*Nk]#select a range of size Nk from all the nodes

    #remember optimalpolicy also returns the argmax
    my_kprime = map(k->optimalpolicy(para,V0,k).kprime,mynodes)
    kprime = MPI.Allgather(my_kprime,comm)#will be a vector for each process 
    #Now get policies
    return Interpoland(kbasis,kprime),V0
end

"""
    getV0(para::NCParameters)

Initializes V0(k) = 0 using the kgrid of para
"""
function getV0(para::NCParameters)
    @unpack kgrid,spline_order = para

    kbasis = Basis(SplineParams(kgrid,0,spline_order))

    return Interpoland(kbasis,k->0 .*k)
end

#Note need to call solvebellman on all processes so they work in 
#parallel
para = NCParameters()
para.kgrid = LinRange(0.05,0.5,102) #102 insures 104 nodes with cubic interpolation
kprime,V = solvebellman_mpi(para,getV0(para))


println(@elapsed solvebellman_mpi(para,getV0(para)))