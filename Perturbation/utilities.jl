using LinearAlgebra
using TensorOperations

import Base.*,LinearAlgebra.⋅

function *(A::Array{T1,3},B::Array{T2}) where {T1<:Real,T2<:Real}
    sA,sB = size(A),size(B)
    return reshape(reshape(A,:,sA[end])*reshape(B,sB[1],:),sA[1:end-1]...,sB[2:end]...)
end

function *(A::Array{T1,4},B::Array{T2}) where {T1<:Real,T2<:Real}
    sA,sB = size(A),size(B)
    return reshape(reshape(A,:,sA[end])*reshape(B,sB[1],:),sA[1:end-1]...,sB[2:end]...)
end

function *(A::Array{T1},B::Array{T2,3}) where {T1<:Real,T2<:Real}
    sA,sB = size(A),size(B)
    return reshape(reshape(A,:,sA[end])*reshape(B,sB[1],:),sA[1:end-1]...,sB[2:end]...)
end

function *(A::Array{T1},B::Array{T2,4}) where {T1<:Real,T2<:Real}
    sA,sB = size(A),size(B)
    return reshape(reshape(A,:,sA[end])*reshape(B,sB[1],:),sA[1:end-1]...,sB[2:end]...)
end



function ⋅(A::Array{T1,3},B::Tuple{Vector{T2},Vector{T3}}) where {T1<:Real,T2<:Real,T3<:Real}
    sA = size(A)
    return reshape(reshape(A,:,sA[3])*B[2],:,sA[2])*B[1]
end

function ⋅(A::Array{T1,3},B::Tuple{Matrix{T2},Matrix{T3}}) where {T1<:Real,T2<:Real,T3<:Real}
    sA,sB1,sB2 = size(A),size(B[1]),size(B[2])
    return permutedims(permutedims(A,[1,3,2])*B[1],[1,3,2])*B[2]
end
#function ⋅(A::Array{T1,3},B::Tuple{Matrix{T2},Matrix{T3}}) where {T1<:Real,T2<:Real,T3<:Real}
#    return @tensor C[i,l,m] := A[i,j,k]*B[1][j,l]*B[2][k,m]
#end

function ⋅(A::Array{T1,3},B::Tuple{UniformScaling{Bool},Matrix{T2}}) where {T1<:Real,T2<:Real}
    return @tensor C[i,j,m] := A[i,j,k]*B[2][k,m]
end

function ⋅(A::Array{T1,3},B::Tuple{Matrix{T2},UniformScaling{Bool}}) where {T1<:Real,T2<:Real}
    return @tensor C[i,m,k] := A[i,j,k]*B[1][j,m]
end

function ⋅(A::Array{T,3},B::Tuple{UniformScaling{Bool},UniformScaling{Bool}}) where {T<:Real}
    return A
end



function test(a::Vector{Union{Float64,Matrix{Float64}}})
    return a[1]*rand(3)+a[2]*rand(3)
end