%James Allen
%Numerical Optimization Homework 2
clc, clear all, close all

% dat=xlsread('C:\Users\Bob\Google Drive\Grad School\Numerical
% Optimization\Homework\MAT258A\Homework_2\binary.xlsx','binary','A2:D401');

% save ('admissions.mat','dat')


alpha=0.00000001;
max_iter=1000;
tol=1e-3;


load('admissions.mat') %Load imported data

y=dat(:,1); %Get y values
u=dat(:,2:3); %get u values



x=zeros(3,max_iter); %initilize x = [a1 a2 beta]'
n_grad=NaN(1,max_iter);


syms a1 a2 b1 y1 u1 u2
L        =y1*(a1*u1+a2*u2+b1)-...
    (1-y1)*log(1+exp(a1*u1+a2+u2+b1));

grad_L=[...
    diff(L,a1,1)
    diff(L,a2,1)
    diff(L,b1,1)];

f_L=matlabFunction(L);
f_grad_L=matlabFunction(grad_L);


% % % % Check of gradient of L %
% % % eps=0.000001;
% % % data_pnt=1;
% % % var=[1 1 1];
% % %
% % % grad_L_act=f_grad_L(var(1),var(2),var(3),u(data_pnt,1),u(data_pnt,2),y(data_pnt));
% % % grad_L_approx=[...
% % %  (f_L(var(1)+eps,var(2),var(3),u(data_pnt,1),u(data_pnt,2),y(data_pnt))-f_L(var(1),var(2),var(3),u(data_pnt,1),u(data_pnt,2),y(data_pnt)))/eps
% % %  (f_L(var(1),var(2)+eps,var(3),u(data_pnt,1),u(data_pnt,2),y(data_pnt))-f_L(var(1),var(2),var(3),u(data_pnt,1),u(data_pnt,2),y(data_pnt)))/eps
% % %  (f_L(var(1),var(2),var(3)+eps,u(data_pnt,1),u(data_pnt,2),y(data_pnt))-f_L(var(1),var(2),var(3),u(data_pnt,1),u(data_pnt,2),y(data_pnt)))/eps];
% % %
% % % disp('  Actual    Approx')
% % % disp([grad_L_act grad_L_approx])

iter_stop=max_iter;
for j=1:max_iter
    
    dx=[0;0;0];
    for k=1:length(y)
        dx=dx+f_grad_L(x(1,j),x(2,j),x(3,j),u(k,1),u(k,2),y(k));
        
    end
    n_grad(j)=norm(dx);
    x(:,j+1)=x(:,j)+alpha.*dx;
    
    if j>2 && abs(n_grad(j)-n_grad(j-1)) < tol
        fprintf('Stalled on iteration %.0f, grad decreased < %.3e over last iteration\n',j,tol)
        x_out=x(:,j+1);
        iter_stop=j;
        break
    end
    
end

x1_val=@(x2) (1/2 - x_out(3)-x_out(2)*x2)/x_out(1);


figure()

for i=1:length(y)
    if y(i)==0
        plot(u(i,2),u(i,1),'ro')
        hold on
    else
        plot(u(i,2),u(i,1),'k+')
        hold on
    end
end

plot(min(u(:,2)):0.1:max(u(:,2)),x1_val(min(u(:,2)):0.1:max(u(:,2))),'-k')

xlabel('GPA')
ylabel('GRE')

%-------------------------------------------------------------------
%This is for newtons method

alpha_new=0.01;

x_new=zeros(3,max_iter); %initilize x = [a1 a2 beta]'
n_grad_new=NaN(1,max_iter);

hes_L=hessian(L,[a1 a2 b1]);

f_hes_L=matlabFunction(hes_L);
n_test=0;

iter_stop_new=max_iter;
for j=1:max_iter
    
    g=[0;0;0];
    h=zeros(3,3);
    for k=1:length(y)
        g=g+f_grad_L(x_new(1,j),x_new(2,j),x_new(3,j),u(k,1),u(k,2),y(k));
        h=h+f_hes_L (x_new(1,j),x_new(2,j),x_new(3,j),u(k,1),u(k,2),y(k));
    end
    n_grad_new(j)=norm(g);
    x_new(:,j+1)=x_new(:,j)+alpha_new*inv(h)*(-g);
    
    if j>2 && abs(n_grad_new(j)-n_grad_new(j-1)) < tol
        fprintf('Stalled on iteration %.0f, grad decreased < %.3e over last iteration\n',j,tol)
        x_out_new=x_new(:,j+1);
        iter_stop_new=j;
        break
    end
    
    if isnan(n_grad_new(j))
        disp('NaN reached for gradient in newtons method aborting. . .')
        n_test=1;
        break
    end
    
end

if n_test==0
    x1_val_new=@(x2) (1/2 - x_out_new(3)-x_out_new(2)*x2)/x_out_new(1);
    
    plot(min(u(:,2)):0.1:max(u(:,2)),x1_val_new(min(u(:,2)):0.1:max(u(:,2))),'--k')
end


figure()

if n_test==0
    plot(1:iter_stop,n_grad(1:iter_stop),'-k',1:iter_stop_new,n_grad_new(1:iter_stop_new),'--k')
    xlabel('iterations')
    ylabel('norm of gradient')
    legend('Steepest Decent','Newtons Method')
else
    plot(1:iter_stop,n_grad(1:iter_stop),'-k')
    xlabel('iterations')
    ylabel('norm of gradient')
    legend('Steepest Decent')
end
%%
% % I think something is wrong with grad_L_a2? ? ?
%initilize grad L
% grad_L_a1=@(y,u,a,b) y(1)*u(1)-(u(1)*(1-y(1))*exp(a(1)*u(1)+a(2)*u(2)+b(1)))/(exp(a(1)*u(1)+a(2)*u(2)+b(1))+1);
% grad_L_a2=@(y,u,a,b) y(1)*u(2)-(u(2)*(1-y(1))*exp(a(1)*u(1)+a(2)*u(2)+b(1)))/(exp(a(1)*u(1)+a(2)*u(2)+b(1))+1);
% grad_L_a3=@(y,u,a,b) y(1)     -(     (1-y(1))*exp(a(1)*u(1)+a(2)*u(2)+b(1)))/(exp(a(1)*u(1)+a(2)*u(2)+b(1))+1);

% L        =@(y,u,a,b) y(1)*(a(1)*u(1)+a(2)*u(2)+b(1))-...
%           (1-y(1))*log(1+exp(a(1)*u(1)+a(2)+u(2)+b(1)));


% subplot(2,1,1)
% plot(1:max_iter+1,x(1,:),1:max_iter+1,x(2,:))
% legend('a_1','a_2')
%
%
% subplot(2,1,2)
% plot(1:max_iter+1,x(3,:))
% ylabel('\beta')
% xlabel('iteration')





