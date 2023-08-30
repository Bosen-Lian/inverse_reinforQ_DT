clear all;
close all;
clc;


%系统
A=[0.906488 0.0816012 -0.0005;
    0.074349 0.90121 -0.000708383;
    0 0 0.132655];

B=[-0.00150808;
    -0.0096;
    0.867345];

D=[0.00951892;
    0.00038373;
    0];



%专家参数
Qe=1*[1 0 0 ;
      0 1 0 ;
      0 0 1 ];
Re=1;
gammae=5;
Pe=dare(A,[B,D],Qe,[Re 0; 0, -gammae^2]);
Ke=inv(Re+B'*Pe*B+B'*Pe*D*inv(gammae^2-D'*Pe*D)*D'*Pe*B)*(B'*Pe*A+B'*Pe*D*inv(gammae^2-D'*Pe*D)*D'*Pe*A);
Le=inv(-gammae^2+D'*Pe*D-D'*Pe*B*inv(Re+B'*Pe*B)*B'*Pe*D)*(D'*Pe*A-D'*Pe*B*inv(Re+B'*Pe*B)*B'*Pe*A);



%学生参数
Q=0*[0.01 0 0 ;
      0 0.01 0 ;
      0 0 0.01];
R=2;
gamma=10;
K1=[1.5    2    1.1];
L1=[0 0 0];
K=K1;
L=L1;



%迭代初值
x(1:3,1)=[1;-1;-1];
xx(1:3,1)=[1;1;-1];
k=0;
kk=0;
ii=0;
jj=0;
eK=[];
eL=[];
eQ=[];
stopq=0;
alpha=0.7; %0.75

for i=1:10000
    
    %专家数据
    pe(:,i)=0.001*(rand(1));
    ue(:,i)=-Ke*x(:,i)+pe(:,i);
%       we(:,i)=-Le*x(:,i)+0.010*(rand(1));
      we(:,i)=0.0010*(rand(1));
    x(:,i+1)=A*x(:,i)+B*ue(:,i)+D*we(:,i);
    
      uue(:,i)=-Ke*xx(:,i);
%         wwe(:,i)=-Le*xx(:,i)+0.0000*(rand(1));
%         wwe(:,i)=0.0010*(rand(1));
    xx(:,i+1)=A*xx(:,i)+B*uue(:,i)+D*we(:,i);  
    
    
    
    ii=ii+1;
    Hxx(ii,:)=[x(1,i)^2, 2*x(1,i)*x(2,i), 2*x(1,i)*x(3,i), x(2,i)^2, 2*x(2,i)*x(3,i),x(3,i)^2]...
              -[x(1,i+1)^2, 2*x(1,i+1)*x(2,i+1), 2*x(1,i+1)*x(3,i+1), x(2,i+1)^2, 2*x(2,i+1)*x(3,i+1),x(3,i+1)^2];%1-6  
    Hxu(ii,:)=2*kron(x(:,i)'*K1',x(:,i)')+2*kron(ue(:,i)',x(:,i)');%7-9
    Hxw(ii,:)=2*kron(x(:,i)'*L1',x(:,i)')+2*kron(we(:,i)',x(:,i)');%10-12
    Huu(ii,:)=kron(ue(:,i)-K1*x(:,i),ue(:,i)+K1*x(:,i));%13
    Huw(ii,:)=kron(ue(:,i)+K1*x(:,i),we(:,i)-L1*x(:,i))+kron(ue(:,i)-K1*x(:,i),we(:,i)+L1*x(:,i));%14
    Hww(ii,:)=kron(we(:,i)-L1*x(:,i),we(:,i)+L1*x(:,i));%15   
    r(ii)=x(:,i)'*K1'*R*K1*x(:,i)+x(:,i)'*Q*x(:,i)-gamma^2*(we(:,i)-L1*x(:,i))'*(we(:,i)+L1*x(:,i))-gamma^2*x(:,i)'*(L1'*L1)*x(:,i)...
        +alpha*(ue(:,i)+K1*x(:,i))'*R*(ue(:,i)+K1*x(:,i))+(ue(:,i)-K1*x(:,i))'*R*(ue(:,i)+K1*x(:,i));

       Hqxx(ii,:)=[xx(1,i)^2, 2*xx(1,i)*xx(2,i), 2*xx(1,i)*xx(3,i), xx(2,i)^2, 2*xx(2,i)*xx(3,i),xx(3,i)^2]...
              -[xx(1,i+1)^2, 2*xx(1,i+1)*xx(2,i+1), 2*xx(1,i+1)*xx(3,i+1), xx(2,i+1)^2, 2*xx(2,i+1)*xx(3,i+1),xx(3,i+1)^2];%1-6  
%         
        Hxuu(ii,:)=2*kron(xx(:,i)'*K1',xx(:,i)')+2*kron(uue(:,i)',xx(:,i)');%7-9  
        Hxww(ii,:)=2*kron(xx(:,i)'*L1',xx(:,i)')+2*kron(we(:,i)',xx(:,i)');%10-12
        Huuu(ii,:)=kron(uue(:,i)-K1*xx(:,i),uue(:,i)+K1*xx(:,i));%13
        Huww(ii,:)=kron(uue(:,i)+K1*xx(:,i),we(:,i)-L1*xx(:,i))+kron(uue(:,i)-K1*xx(:,i),we(:,i)+L1*xx(:,i));%14
        Hwww(ii,:)=kron(we(:,i)-L1*xx(:,i),we(:,i)+L1*xx(:,i));%15   
       n(ii)=-xx(:,i)'*K1'*R*K1*x(:,i)+gamma^2*(we(:,i)-L1*xx(:,i))'*(we(:,i)+L1*xx(:,i))...
              -(uue(:,i)-K1*xx(:,i))'*R*(K1*xx(:,i)+uue(:,i))+gamma^2*xx(:,i)'*(L1'*L1)*xx(:,i);
%          n(ii)=-xx(:,i)'*K1'*R*K1*x(:,i)+gamma^2*(wwe(:,i)-L1*xx(:,i))'*(wwe(:,i)+L1*xx(:,i))...
%                +(K1*xx(:,i)-uue(:,i))'*R*(K1*xx(:,i)+uue(:,i))+gamma^2*xx(:,i)'*(L1'*L1)*xx(:,i);
%            n(ii)=xx(:,i)'*Q*xx(:,i)+alpha*(uue(:,i)+K1*xx(:,i))'*R*(uue(:,i)+K1*xx(:,i));

    
    
    
    jj=jj+1;
    if stopq==0
      
       X(jj,:)=[xx(1,i)^2, 2*xx(1,i)*xx(2,i), 2*xx(1,i)*xx(3,i), xx(2,i)^2, 2*xx(2,i)*xx(3,i),xx(3,i)^2];


%           
       Hqxxx(jj,:)=Hqxx(ii,:)
       Hqxu(jj,:)=Hxuu(ii,:);
       Hqxw(jj,:)=Hxww(ii,:);
       Hquu(jj,:)=Huuu(ii,:);
       Hquw(jj,:)=Huww(ii,:);
       Hqww(jj,:)=Hwww(ii,:);
         rq(jj)= n(ii);


    end
    
    rank(X)
    
    if rank(X)==6
        stopq=1;
    end
    
    zbar=[Hxx Hxu Hxw Huu Huw Hww];
%     m=zbar'*zbar; 
    m=zbar; 
%     q=zbar'*r';
    a=rank(m)
        
    if a==15
        
       H=m\r';

       K1=inv(H(13)-H(14)*inv(H(15))*H(14))*([H(7) H(8) H(9)]-H(14)*inv(H(15))*[H(10) H(11) H(12)]);
       L1=inv(H(15)-H(14)*inv(H(13))*H(14))*([H(10) H(11) H(12)]-H(14)*inv(H(13))*[H(7,1) H(8,1) H(9,1)]);
       
       K=[K;K1];
       L=[L,L1];
       
       eK=[eK,norm(K1-Ke)];
       eL=[eL,norm(L1-Le)];    

        Hxx=[];
        Hxu=[];
        Hxw=[];
        Huu=[];
        Huw=[];
        Hww=[];
        zbar=[];
        r=[];
        ii=0;
        m=[];
        
%         Q=Q+alpha*(Ke-K1)'*R*(Ke-K1);
%            eQ=[eQ,norm(Q)];
          Hq=Hqxxx*[H(1),H(2),H(3),H(4),H(5),H(6)]'...
              +Hqxu*[H(7) H(8) H(9)]'+Hqxw*[H(10) H(11) H(12)]'+Hquu*H(13)+Hquw*H(14)+Hqww*H(15)+rq;
 

         Hq=rq;
        HQ=X\Hq';
        Q=[HQ(1),HQ(2),HQ(3);
           HQ(2),HQ(4),HQ(5);
           HQ(3),HQ(5),HQ(6)];
        eQ=[eQ,norm(Q)];
        jj=0;
        stopq=0;
        X=[];
        Hqxxx=[];
        Hqxu=[];
        Hqxw=[];
        Hquu=[];
        Hquw=[];
        Hqww=[];
        Hq=[];
        
    end
    


end
 
figure(1)
plot(eQ,'-*','LineWidth',1)
legend('Q')

figure(2)
plot(eK,'-*','LineWidth',1)
legend('K-Ke')


figure(3)
plot(eL,'-*','LineWidth',1)
legend('L-Le')

xt(1:3,1)=[1;-1;-1];
xxt(1:3,1)=[1;-1;-1];


for j=1:60
    
       ute(:,j)=-Ke*xt(:,j);
      wte(:,j)=0.0010*(rand(1));
    xt(:,j+1)=A*xt(:,j)+B*ute(:,j)+D*wte(:,j); 
    
          uute(:,j)=-K1*xxt(:,j);
      wwte(:,j)=0.0010*(rand(1));
    xxt(:,j+1)=A*xxt(:,j)+B*uute(:,j)+D*wte(:,j);  
    
    
end

j=0:1:60;
plot(j,xt(1,:),j,xt(2,:),j,xt(3,:),'LineWidth',1);
hold on 
plot(ute(:),'LineWidth',1);
hold on
plot(j,xxt(1,:),':',j,xxt(2,:),'-',j,xxt(3,:),'-.','LineWidth',1);
hold on 
plot(uute(:),'LineWidth',1);




