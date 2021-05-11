using FastGaussQuadrature

ξx,wx = gausslegendre(2)
ξy,wy = gausslegendre(3)

ξ = hcat(kron(ones(length(ξy)),ξx),kron(ξy,ones(length(ξx))))
w = kron(wy,wx)

plot(x=ξ[:,1],y=ξ[:,2])

ξ = hcat([[ξx[i],ξy[j]] for i in 1:2 for j in 1:3]...)'


D = length(N)
#dimension 1
ξ,W = gausslegendre(N[1])
X = ξ.*(b[1]-a[1])/2 .+ b[1]/2 .+ a[1]/2 #X is going to be an N^D×D vector

for i in 2:D
    ξ,w = gausslegendre(N[i])
    #Next rescale nodes to be on [a,b]
    x = ξ.*(b[i]-a[i])/2 .+ b[i]/2 .+ a[i]/2

    X = hcat(kron(ones(N[i]),X),kron(x,ones(size(X,1))))
    W = kron(w,W)
end