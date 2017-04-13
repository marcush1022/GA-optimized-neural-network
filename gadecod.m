function[W1,B1,W2,B2,val]=gadecod(x)
global p
global t
global R
global S2
global S1
global S

% First R*S1(intput*hidden) CODE is W1
for i=1:S1
    for k=1:R
        W1(i,k)=x(R*(i-1)+k);
    end
end

% Next S1*S2(hidden*output, after R*S1) CODE is w2 
for i=1:S2
    for k=1:S1
        W2(i,k)=x(S1*(i-1)+k+R*S1);
    end
end

% Next S1(hidden, after R*S1+S1*S2) CODE is B1
for i=1:S1
    B1(i,1)=x((R*S1+S1*S2)+i);
end

% Next S2(output, after R*S1+S1*S2+S1) CODE is B2
for i=1:S2
    B2(i,1)=x((R*S1+S1*S2+S1)+i);
end

% Calcu the output of layer S1 and S2 
A1=tansig(W1*p,B1);
A2=purelin(W2*A1,B2);

% Calcu the Sum-Square-Error
SE=sumsqr(t-A2);

% Calcu the fitness of GA
val=1/SE;



