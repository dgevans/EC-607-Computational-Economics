"""
    montecarlo_integration(f,a,b,N)

Computes integral using Monte Carlo methods
"""
function montecarlo_integration(f,a::Vector,b::Vector,N)
    D = length(a)
    X = rand(D,N).*(b.-a) .+ a
    #return weighted sum (note need to scale based on (b-a))
    return sum(f(X[:,i]) for i in 1:N)/N*prod(b.-a)
end;
f_runge_ND(x) = 1 /(1 + 25 * norm(x)) #The Runge Function


using DataFrames,LinearAlgebra
Nmax = 10^5
dfMonte = DataFrame()
dfMonte.N = 1000:10000:Nmax
for D in 1:10
    dfMonte[!,"dimension$D"] = [montecarlo_integration(f_runge_ND,-1*ones(D),ones(D),N) for N in 1000:10000:Nmax]
    dfMonte[!,"dimension$D"] ./= dfMonte[end,"dimension$D"] #normalize by last point
end

using QuasiMonteCarlo

