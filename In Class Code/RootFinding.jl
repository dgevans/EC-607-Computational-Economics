using ForwardDiff


f(x) = x^2 - 2 + sin(x)
df(x) = 2*x + cos(x)

f(2)

dfhat(x) = ForwardDiff.derivative(f,x)
df(2)
dfhat(2)