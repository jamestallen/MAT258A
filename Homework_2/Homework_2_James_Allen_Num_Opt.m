%James Allen
%Numerical Optimization Homework 2
clc, clear all

% dat=xlsread('C:\Users\Bob\Google Drive\Grad School\Numerical
% Optimization\Homework\MAT258A\Homework_2\binary.xlsx','binary','A2:D401');

% save ('admissions.mat','dat')

load('admissions.mat') %Load imported data

y=dat(:,1); %Get y values
u=dat(:,2:3); %get u values


%initilize x = [a beta]'
x=ones(2,length(y));

%initilize grad L
grad_L=@(y,u,a,b) y.*(u+b)-(1-y).*(exp(a'.*u+b)).*(exp(a'.*u+b)+1).^-1.*(u+1);

alpha=0.001;

%To test the grad_L function. . . it does not work, the very first one is
%messed up. . y(i) is a scalar but u is not. . . 

% w=1;
% (y(w).*(u(w,:)+x(2,w))-(1-y(w)).*(exp(x(1,w).*u(w,:)+x(2,w))).*(exp(x(1,w).*u(w,:)+x(2,w))+1).^-1.*(u(w,:)+1))'


for k=1:length(y)
    x(:,k+1)=x(:,k)+alpha*grad_L(y(k),u(k,:),x(1,k),x(2,k))';
end






