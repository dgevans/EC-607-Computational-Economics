println("I have $(Threads.nthreads()) active threads")
println("This is my id $(Threads.threadid())")#Main Thread id is 1

a = zeros(10)
for i in 1:10
    a[i] = i^2
end
println(a)

a = zeros(10)
Threads.@threads for i in 1:10
    a[i] = i^2
end
println(a)

a = zeros(10)
id = zeros(10)
Threads.@threads for i in 1:10
    a[i] = i^2
    id[i] = Threads.threadid()
end
println(a)
println(id)

acc = zeros(3)
for i in 1:10_000
    acc .+= 1
end
println(acc)

acc = zeros(3)
Threads.@threads for i in 1:10_000
    acc .+= 1
end
println(acc)

using Parameters
mutable struct AR1
    μ::Float64 #Mean of the AR(1)
    ρ::Float64 #persistence of the AR(1)
    σ::Float64 #standard deviaiton of the AR(1)
end
"""
   simulateAR1(ar,x0,T)

Simulates an AR(1) ar for T periods with initial value x0
"""
function simulateAR1(ar,x0,T)
    @unpack σ,μ,ρ = ar #note order doesn't matter
    x = zeros(T+1)# initialize
    x[1] = x0
    for t in 1:T
        x[t+1] = (1-ρ)*μ + ρ*x[t] + σ*randn()
    end
    return x[2:end]
end;

#get a sample of 100_000 of x after 100 periods
"""
    draw_endofT_sample(ar,N,T)

Draws N values of x after T periods where x follows AR1 with x[1] = 0.
"""
function draw_endofT_sample(ar,N,T)
    sample = zeros(N)
    for i in 1:N
        sample[i] = simulateAR1(ar,0.,T)[end]
    end
    return sample
end

ar = AR1(1.,.99,0.01)
sample = draw_endofT_sample(ar,100_000,1000)
@time draw_endofT_sample(ar,100_000,1000);

"""
    draw_endofT_sample(ar,N,T)

Draws N values of x after T periods where x follows AR1 with x[1] = 0.
"""
function draw_endofT_sample_parallel(ar,N,T)
    sample = zeros(N)
    Threads.@threads for i in 1:N
        sample[i] = simulateAR1(ar,0.,T)[end]
    end
    return sample
end

sample = draw_endofT_sample_parallel(ar,100_000,1000)
@time draw_endofT_sample_parallel(ar,100_000,1000);

"""
   simulateAR1final(ar,x0,T)

Simulates an AR(1) ar for T periods with initial value x0.
Returns the final period value
"""
function simulateAR1final(ar,x,T)
    @unpack σ,μ,ρ = ar #note order doesn't matter
    for t in 1:T
        x = (1-ρ)*μ + ρ*x + σ*randn()
    end
    return x
end

"""
    draw_endofT_sample(ar,N,T)

Draws N values of x after T periods where x follows AR1 with x[1] = 0.
"""
function draw_endofT_sample(ar,N,T)
    sample = zeros(N)
    for i in 1:N
        sample[i] = simulateAR1final(ar,0.,T)
    end
    return sample
end
sample = draw_endofT_sample(ar,100_000,1000)
@time draw_endofT_sample(ar,100_000,1000);

"""
    draw_endofT_sample(ar,N,T)

Draws N values of x after T periods where x follows AR1 with x[1] = 0.
"""
function draw_endofT_sample_parallel(ar,N,T)
    sample = zeros(N)
    Threads.@threads for i in 1:N
        sample[i] = simulateAR1final(ar,0.,T)
    end
    return sample
end
sample = draw_endofT_sample_parallel(ar,100_000,1000)
@time draw_endofT_sample_parallel(ar,100_000,1000);

using BasisMatrices,Optim,LinearAlgebra
@with_kw mutable struct NCParameters
    A::Float64 = 1.
    α::Float64 = 0.3
    β::Float64 = 0.96
    kgrid::Vector{Float64} = LinRange(0.05,0.5,20)
    spline_order::Int = 3
end;

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
end;

"""
    bellmanmap(Vprime,para::NCParameters)

Apply the bellman map given continuation value function Vprime
"""
function bellmanmap(para::NCParameters,Vprime::Interpoland)
    kbasis = Vprime.basis
    #sometimes it's helpful to tell julia what type a variable is
    knodes = nodes(kbasis)[1]::Vector{Float64}
    V = map(k->optimalpolicy(para,Vprime,k).V,knodes)
    return Interpoland(kbasis,V)
end

"""
    solvebellman(para::NCParameters,V0::Interpoland)

Solves the bellman equation for a given V0
"""
function solvebellman(para::NCParameters,V0::Interpoland)
    diff = 1
    #Iterate of Bellman Map until difference in coefficients goes to zero
    while diff > 1e-6
        V = bellmanmap(para,V0)
        diff = norm(V.coefs-V0.coefs,Inf)
        V0 = V
    end
    kbasis = V0.basis
    knodes = nodes(kbasis)[1]::Vector{Float64}
    #remember optimalpolicy also returns the argmax
    kprime = map(k->optimalpolicy(para,V0,k).kprime,knodes)
    #Now get policies
    return Interpoland(kbasis,kprime),V0
end;

"""
    getV0(para::NCParameters)

Initializes V0(k) = 0 using the kgrid of para
"""
function getV0(para::NCParameters)
    @unpack kgrid,spline_order = para

    kbasis = Basis(SplineParams(kgrid,0,spline_order))

    return Interpoland(kbasis,k->0 .*k)
end
para = NCParameters()
para.kgrid = LinRange(0.05,0.5,100)
kprime,V = solvebellman(para,getV0(para))
@time solvebellman(para,getV0(para));

using ThreadTools
"""
    bellmanmap_parallel(Vprime,para::NCParameters)

Apply the bellman map given continuation value function Vprime
"""
function bellmanmap_parallel(para::NCParameters,Vprime::Interpoland)
    kbasis = Vprime.basis
    #sometimes it's helpful to tell julia what type a variable is
    knodes = nodes(kbasis)[1]::Vector{Float64}
    V = tmap(k->optimalpolicy(para,Vprime,k).V,knodes)
    return Interpoland(kbasis,V)
end

"""
    solvebellman_parallel(para::NCParameters,V0::Interpoland)

Solves the bellman equation for a given V0
"""
function solvebellman_parallel(para::NCParameters,V0::Interpoland)
    diff = 1
    #Iterate of Bellman Map until difference in coefficients goes to zero
    while diff > 1e-6
        V = bellmanmap_parallel(para,V0)
        diff = norm(V.coefs-V0.coefs,Inf)
        V0 = V
    end
    kbasis = V0.basis
    knodes = nodes(kbasis)[1]::Vector{Float64}
    #remember optimalpolicy also returns the argmax
    kprime = tmap(k->optimalpolicy(para,V0,k).kprime,knodes)
    #Now get policies
    return Interpoland(kbasis,kprime),V0
end;

kprime,V = solvebellman_parallel(para,getV0(para))
@time solvebellman_parallel(para,getV0(para));

using MPIClusterManagers,Distributed
manager = MPIManager(np=4)

#spawns the additional processes
addprocs(manager)

@mpi_do manager begin #manager represents the cluster we are working with
    a = 2
    b = 3
    println("For a^b I get $(a^b)")
end;

@mpi_do manager begin
    using MPI
    comm=MPI.COMM_WORLD #comm defines who is communicating, comm_world is all the processes
    println("Hello world, I am $(MPI.Comm_rank(comm)) of $(MPI.Comm_size(comm))")
end

@mpi_do manager begin
    data = rand(1)
    println("Hi, I'm rank $(MPI.Comm_rank(comm)) and my data is $data")
    MPI.Bcast!(data,0,comm) #Bcast data using root 0 and COMM_WORLD communicator
    println("Hi, I'm rank $(MPI.Comm_rank(comm)) and my data is now  $data")
end

@mpi_do manager begin
    data = rand(2)
    println("Hi, I'm rank $(MPI.Comm_rank(comm)) and my data is $data")
    gathered_data = MPI.Gather(data,0,comm) #Bcast data using root 0 and COMM_WORLD communicator
    println("Hi, I'm rank $(MPI.Comm_rank(comm)) here is the combined data  $gathered_data")
end

@mpi_do manager begin
    data = rand(1)
    println("Hi, I'm rank $(MPI.Comm_rank(comm)) and my data is $data")
    gathered_data = MPI.Allgather(data,comm) #Bcast data using root 0 and COMM_WORLD communicator
    println("Hi, I'm rank $(MPI.Comm_rank(comm)) here is the combined data  $gathered_data")
end

@mpi_do manager begin
    using Parameters
end

@mpi_do manager begin
    using Parameters
    mutable struct AR1
        μ::Float64 #Mean of the AR(1)
        ρ::Float64 #persistence of the AR(1)
        σ::Float64 #standard deviaiton of the AR(1)
    end


    """
    simulateAR1(ar,x0,T)

    Simulates an AR(1) ar for T periods with initial value x0
    """
    function simulateAR1(ar,x0,T)
        @unpack σ,μ,ρ = ar #note order doesn't matter
        x = zeros(T+1)# initialize
        x[1] = x0
        for t in 1:T
            x[t+1] = (1-ρ)*μ + ρ*x[t] + σ*randn()
        end
        return x[2:end]
    end


    #get a sample of 100_000 of x after 100 periods
    """
        draw_endofT_sample(ar,N,T)

    Draws N values of x after T periods where x follows AR1 with x[1] = 0.
    """
    function draw_endofT_sample(ar,N,T)
        sample = zeros(N)
        for i in 1:N
            sample[i] = simulateAR1(ar,0.,T)[end]
        end
        return sample
    end
end;

@mpi_do manager begin 
    ar = AR1(1.,.99,0.01)
    sample = draw_endofT_sample(ar,50000,1000) #note each process draws its own sample 
    #How to get them all together
end;
@fetchfrom 5 sample

@mpi_do manager begin
    using Statistics
    samples = MPI.Allgather(sample,comm) #All the processes gather data
    println(std(samples))
end

@mpi_do manager begin
    using BasisMatrices,Optim,LinearAlgebra
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
        getV0(para::NCParameters)

    Initializes V0(k) = 0 using the kgrid of para
    """
    function getV0(para::NCParameters)
        @unpack kgrid,spline_order = para

        kbasis = Basis(SplineParams(kgrid,0,spline_order))

        return Interpoland(kbasis,k->0 .*k)
    end
end;

@mpi_do manager begin
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
end;

@mpi_do manager begin    
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
end

@mpi_do manager begin
    #Note need to call solvebellman on all processes so they work in 
    #parallel
    para = NCParameters()
    para.kgrid = LinRange(0.05,0.5,102) #102 insures 104 nodes with cubic interpolation
    kprime,V = solvebellman_mpi(para,getV0(para))
end
@time @mpi_do manager begin
    solvebellman_mpi(para,getV0(para))
end