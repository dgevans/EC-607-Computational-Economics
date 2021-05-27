using Gadfly

a = zeros(10)
id = zeros(10)
for i in 1:10
    a[i] = i^2
    id[i] = Threads.threadid()
end
println(a)
println(id)


using MPIClusterManagers,Distributed
manager = MPIManager(np=4)

#spawns the additional processes
addprocs(manager)

@mpi_do manager begin
    using MPI
    comm = MPI.COMM_WORLD
    r = MPI.Comm_rank(comm)
    s = MPI.Comm_size(comm)
end

@mpi_do manager begin
    data = rand(2)
    println("Hi, I'm rank $(MPI.Comm_rank(comm)) and my data is $data")
    gathered_data = MPI.Gather(data,0,comm) #Bcast data using root 0 and COMM_WORLD communicator
    println("Hi, I'm rank $(MPI.Comm_rank(comm)) here is the combined data  $gathered_data")
end