% main: ga_bp.m
% fitness function: gabpEval.m
% gadecod.m
clear all
clc
warning off 
nntwarn off
%%%%%%%%%%%%%%%%%%%%%% Global Parameters %%%%%%%%%%%%%%%%%%%%%%%
global p     % training set input data
global t     % training set output data
global R     % numbers of input layer neruons 
global S2    % numbers of output layer neruons
global S1    % numbers of hidden layer neruons
global S     % length of coding?
S1=25;

%%%%%%%%%%%%%%%%%%% Input Data %%%%%%%%%%%%%%%%%%%%%%%%%%% 

% training data set
day=[0.9363 -0.9698 -0.9907 -0.9562 -0.9507 0.9363 -0.9164 0.9045 0.8918;
 -0.9358 -0.9751 0.9821 -0.9544 -0.9469 0.9426 0.9182 0.8967 -0.8841;
0.9516 -0.9781 -0.9744 -0.9525 0.9509 0.9368 0.9082 -0.8903 -0.8665;
 -0.9480 -0.9795 -0.9796 -0.9507 0.9509 0.9300 -0.9075 -0.8902 -0.8671;
 -0.9433 -0.9923 -0.9812 -0.9596 -0.9406 -0.9230 0.9071 -0.8864 -0.8547;
 -0.9424 1.0000 -0.9800 -0.9514 0.9349 -0.9089 0.9206 -0.8780 -0.8414;
0.9355 -0.9878 -0.9737 -0.9499 0.9337 0.9084 -0.9072 -0.8745 -0.8332];

% normalization of data
[dayn,minday,maxday]=premnmx(day);

% input set and output set
p=dayn(:,1:8);
t=dayn(:,2:9);

% validation data set
k=[0.9435 0.9796 -0.9706 -0.9552 -0.9298 -0.9130 -0.9003 0.8708 0.8234;
    -0.9358 -0.9751 0.9821 -0.9544 -0.9469 0.9426 0.9182 0.8967 -0.8841;
0.9516 -0.9781 -0.9744 -0.9525 0.9509 0.9368 0.9082 -0.8903 -0.8665;
 -0.9480 -0.9795 -0.9796 -0.9507 0.9509 0.9300 -0.9075 -0.8902 -0.8671;
 -0.9433 -0.9923 -0.9812 -0.9596 -0.9406 -0.9230 0.9071 -0.8864 -0.8547;
 -0.9424 1.0000 -0.9800 -0.9514 0.9349 -0.9089 0.9206 -0.8780 -0.8414;
     -0.9496 -0.9778 -0.9693 -0.9536 -0.9352 -0.9111 -0.9076 0.8797 -0.8227];
 
% normalization of data
 kn=tramnmx(k,minday,maxday);
 
%%%%%%%%%%%%%%%%%%%% BP Nerual Network %%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
% Create Network
net=newff(minmax(p),[S1,7],{'tansig','purelin'},'trainlm'); 

% Set training parameters
net.trainParam.show=10;
net.trainParam.epochs=2000;
net.trainParam.goal=1.0e-28;
net.trainParam.lr=0.3;

% Train network
[net,tr]=train(net,p,t);

% Use the Simulink
s_bp=sim(net,kn);    % Simulink result of BP Network

%%%%%%%%%%%%%%%%%%%%%%% GA-BP Network %%%%%%%%%%%%%%%%%%%%%

R=size(p,1);
S2=size(t,1);
S=R*S1+S1*S2+S1+S2;
aa=ones(S,1)*[-1,1];
popu=50;  % Number of population
initPpp=initializega(popu,aa,'gabpEval');  % initialize of population
gen=100;  % iter time of generatic

% Use the GAOT toolbox, the object function is gabpEval.m
[x,endPop,bPop,trace]=ga(aa,'gabpEval',[],initPpp,[1e-6 1 1],'maxGenTerm',gen,...
'normGeomSelect',[0.09],['arithXover'],[2],'nonUnifMutation',[2 gen 3]);

% Figure: changes of MSE
figure(1)
plot(trace(:,1),1./trace(:,3),'r-');
hold on
plot(trace(:,1),1./trace(:,2),'b-');
xlabel('Generation');
ylabel('Sum-Squared Error');

% Figure: changes of Fittness
figure(2)
plot(trace(:,1),trace(:,3),'r-');
hold on
plot(trace(:,1),trace(:,2),'b-');
xlabel('Generation');
ylabel('Fittness');

% Calcu the best weights and threshold 
[W1,B1,W2,B2,val]=gadecod(x);
net.IW{1,1}=W1;
net.LW{2,1}=W2;
net.b{1}=B1;
net.b{2}=B2;

% Train the network with new weights and bias
net=train(net,p,t);

% Simulink test
s_ga=sim(net,kn);     %Simulink result after optimized by GA

