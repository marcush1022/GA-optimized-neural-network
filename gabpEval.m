function[sol,val]=gabpEval(sol,options)
global S
for i=1:S
x(i)=sol(i);
end;
[W1,B1,W2,B2,val]=gadecod(x);
