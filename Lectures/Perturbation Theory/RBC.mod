var c k z r;
varexo e;
parameters bet del alp sig rho s;

bet = 0.96;
del = 0.1;
alp = 0.3;
sig = 2;
rho = 0.85;
s = 0.015;

model;
(c)^(-sig)=bet*(c(+1)^(-sig)*r(+1));
k=exp(z)*k(-1)^alp-c+(1-del)*k(-1);
r = 1+alp*exp(z)*k(-1)^(alp-1)-del;
z = rho*z(-1)+s*e;
end;

shocks;
var e;
stderr 1;
end;

initval;
k = 1;
c = 1;
z = 0;
e = 0;
end;



steady;

stoch_simul(order = 1, periods=1000);


