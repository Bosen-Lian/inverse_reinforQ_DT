close all;
clc;
clear all;
warning off;

% Inverse RL simulation for DT systems--Wenqian Xue
% data based Agorithm 2
% Demo version



step=1800;

%system dynamics
A=[-1       0.9      1.3
    0.8     -1.1     -0.7
    0       0        -1];
B=[0 0 1]';
C=[1,0,2];

%expert's settings
Rd=2;
Qd=[15,0,0;0,10,0;0,0,10];
[Pd,M,N]=dare(A,B,Qd,Rd);
Kd=(Rd+B'*Pd*B)\(B'*Pd*A);
Hd=[Qd+A'*Pd*A, A'*Pd*B
    B'*Pd*A, Rd+B'*Pd*B];


%expert's data
xd=[10;10;5];
ud=0;
for i=1:step+1  
    ed(i)=0.5*(rand(1));%probing noise
    ud(:,i)=-Kd*xd(:,i)+ed(i);
    xd(:,i+1)=A*xd(:,i)+B*ud(:,i); 
end

%intialization for Algorithm 2
R=1;
Qs=[1,0,0;0,1,0;0,0,1];
P=[5.9722   -1.6774   -9.8598
   -1.6774    10.0167    5.1472
   -9.8598    5.1472    7.8514];
K=(R+B'*P*B)\(B'*P*A);
K=3*[2,-1,-1.8];
Q=Qs;
vQ=[Q(1,1) Q(1,2) Q(1,3) Q(2,2) Q(2,3) Q(3,3)]';
H=zeros(4,4);
x=[10;10;5];

alpha=1;
stopi=0;
stopj=0;
stopl=0;

dQd=[];
dKd=norm(K-Kd);
dHd=[];
dQ=[];
dK=2;
dH=[];

j=1;

dxdud=[];

%learning process
for i=1:step
    

    Qlast=Q(:,:,end);
    Hlast=H(:,:,end);     
    Klast=K(end,:);
    
         
    %collect data
    if(stopi==0)
        
    for s=1:step
        
    zds=[xd(:,s);ud(s)];
    zds1=[xd(:,s+1);ud(:,s+1)-ed(s+1)];
    dxdud(:,s)=[zds(1)^2,2*zds(1)*zds(2),2*zds(1)*zds(3),2*zds(1)*zds(4),...
                zds(2)^2,2*zds(2)*zds(3),2*zds(2)*zds(4),...
                zds(3)^2,2*zds(3)*zds(4),...
                zds(4)^2]'...
               -[zds1(1)^2,2*zds1(1)*zds1(2),2*zds1(1)*zds1(3),2*zds1(1)*zds1(4),...
                 zds1(2)^2,2*zds1(2)*zds1(3),2*zds1(2)*zds1(4),...
                 zds1(3)^2,2*zds1(3)*zds1(4),...
                 zds1(4)^2]';
%             dxdud(:,s) = [zds(1)^2; 2*zds(1)*zds(2:4);...
%               zds(2)^2; 2*zds(2)*zds(3:4);...
%               zds(3)^2; 2*zds(3)*zds(4);...
%                 zds(4)^2] ...
%             -[zds1(1)^2; 2*zds1(1)*zds1(2:4);...
%                zds1(2)^2; 2*zds1(2)*zds1(3:4);...
%                zds1(3)^2; 2*zds1(3)*zds1(4);...
%                  zds1(4)^2];
    rhoxx(:,s) =-(kron(xd(:,s+1),xd(:,s+1))-kron(xd(:,s),xd(:,s)));
    rhoxu(:,s) =-(kron(xd(:,s+1),ud(:,s+1)-ed(:,s+1))-kron(xd(:,s),ud(:,s)));
    rhoux(:,s) =-(kron(ud(:,s+1)-ed(:,s+1),xd(:,s+1))-kron(ud(:,s),xd(:,s)));
    rhouu(:,s) =-(kron(ud(:,s+1)-ed(:,s+1),ud(:,s+1)-ed(:,s+1))-kron(ud(:,s),ud(:,s)));
    rho(:,s)=[rhoxx(1,s);
             rhoxx(2,s)+rhoxx(4,s);
             rhoxx(3,s)+rhoxx(7,s);
             rhoxu(1,s)+rhoux(1,s);
             rhoxx(5,s);
             rhoxx(6,s)+rhoxx(8,s);
             rhoxu(2,s)+rhoux(2,s);
             rhoxx(9,s);
             rhoxu(3,s)+rhoux(3,s);
             rhouu(:,s)];
%     rr(:,s)=[rhoxx(:,s);rhoxu(:,s); rhoux(:,s);rhouu(:,s)];
    qrd(s)=xd(:,s)'*Q(:,:,j)*xd(:,s)+ud(s)'*R*ud(s);
          
    
%     a=rank(dxdud'*dxdud);
    a=rank(dxdud);
    
    if(a==10&&stopj==0)
        
%         vH=(dxdud*dxdud')\dxdud*qrd';
        vH=qrd/dxdud;
        H(:,:,j+1)=[vH(1),vH(2),vH(3),vH(4);
                  vH(2),vH(5),vH(6),vH(7);
                  vH(3),vH(6),vH(8),vH(9);
                  vH(4),vH(7),vH(9),vH(10)];
        dH=[dH,norm(H(:,:,j+1)-Hlast)];
        dHd=[dHd,norm(H(:,:,end)-Hd)];
        Hp(:,:,j+1)=H(:,:,j+1)-[Q(:,:,j),[0;0;0];
                            [0,0,0], R];
        K(j+1,:)=H(4,4,j+1)\H(4,1:3,j+1);
        dK=[dK,norm(K(j+1,:)-Klast)];
        dKd=[dKd,norm(K(end,:)-Kd)];        
          
        dxdud=[];
        qrd=[];
        stopi=1;
    end
    
    if(a==10&&stopj==1)
        H(:,:,j+1)=H(:,:,j);
        Hp(:,:,j+1)=Hp(:,:,j);
        K(j+1,:)=K(j,:);
        dK=[dK,norm(K(j+1,:)-Klast)];
        dH=[dH,norm(H(:,:,j+1)-Hlast)];
        dHd=[dHd,norm(H(:,:,end)-Hd)];
        dKd=[dKd,norm(K(end,:)-Kd)];
        stopi=1;
    end
    
    if(stopi==1)
        break;
    end
    
    end
    end
       
        
    if(stopl==1)
       x(:,i)=[10;10;5];
       w=i;
       stopl=0;
    else 
        w=1;
    end
    
    u(:,i)=-K(j+1,:)*x(:,i);
    x(:,i+1)=A*x(:,i)+B*u(:,i);
    y(i)=C*x(:,i);
    
    xx(:,i)=[x(1,i+1)^2, 2*x(1,i+1)*x(2,i+1), 2*x(1,i+1)*x(3,i+1), x(2,i+1)^2, 2*x(2,i+1)*x(3,i+1), x(3,i+1)^2]';
    qr(i)=(1-alpha)*x(:,i+1)'*Q(:,:,j)*x(:,i+1)+...
           alpha*(-(-K(j+1,:)*x(:,i+1))'*R*(-K(j+1,:)*x(:,i+1))+[x(:,i);-K(j+1,:)*x(:,i)]'*Hp(:,:,j+1)*[x(:,i);-K(j+1,:)*x(:,i)]-[x(:,i+1);-K(j+1,:)*x(:,i+1)]'*Hp(:,:,j+1)*[x(:,i+1);-K(j+1,:)*x(:,i+1)]);
    
%     b=rank(xx*xx')
    b=rank(xx);
    
    
    if(stopi==1&&b==6&&stopj==0)
        
%         vQ=(xx*xx')\xx*qr';
        vQ=qr/xx(:,w:i);
        Q(:,:,j+1)=[vQ(1),vQ(2),vQ(3);
                    vQ(2),vQ(4),vQ(5);
                    vQ(3),vQ(5),vQ(6)];
           

        %store learned results    
        dQ=[dQ,norm(Q(:,:,j+1)-Qlast)];
        dQd=[dQd,norm(Q(:,:,end)-Qd)];

        
        
        j=j+1;
        xx=[];
        qr=[];
        stopi=0;
        stopl=1;
        
        
    end
    
        if(stopi==1&&b==6&&stopj==1)
        
           Q(:,:,j+1)=Q(:,:,j);
           
        %store learned results   
        dQ=[dQ,norm(Q(:,:,j+1)-Qlast)];
        dQd=[dQd,norm(Q(:,:,end)-Qd)];

        j=j+1;
        stopi=0;
        stopl=1;
        end
    
        if(dK(end)<0.0001)
           stopj=1;
          
        end




end

%test trajectory under learned result
step=20;
cK=Klast;
xc=[10;10;5];
xd1=[10;10;5];
for i=1:step   
    uc(:,i)=-cK*xc(:,i);%learner
    xc(:,i+1)=A*xc(:,i)+B*uc(:,i);
    yc(i)=C*xc(:,i);
    ud1(:,i)=-Kd*xd1(:,i);%expert
    xd1(:,i+1)=A*xd1(:,i)+B*ud1(:,i);
    yd1(i)=C*xd1(:,i);
end


T=1;
t=1:step;

figure(1)
subplot(3,1,1)
plot(dHd,'-o','LineWidth',1);
legend('||H^{i+1}-H_T||');
hold on;
subplot(3,1,2)
plot(dQd,'-o','LineWidth',1);
legend('||Q^{i+1}-Q_T||');
hold on;
subplot(3,1,3)
plot(dKd,'-o','LineWidth',1);
legend('||K^{i+1}-K_T||');
xlabel('Update steps');

figure(2)
subplot(3,1,1)
plot(t,xc(1,t),'LineWidth',1);
hold on;
plot(t,xd1(1,t),'--','LineWidth',1);
legend('x_1','x_{T1}');
subplot(3,1,2)
plot(t,xc(2,t),'LineWidth',1);
hold on;
plot(t,xd1(2,t),'--','LineWidth',1);
legend('x_2','x_{T2}');
subplot(3,1,3)
plot(t,xc(3,t),'LineWidth',1);
hold on;
plot(t,xd1(3,t),'--','LineWidth',1);
legend('x_3','x_{T3}');
xlabel('Time(s)')

figure(3)
subplot(2,1,1)
plot(T*t,yc,'LineWidth',1);
hold on;
plot(T*t,yd1,'--','LineWidth',1);
legend('y','y_T');
subplot(2,1,2)
plot(T*t,uc,'LineWidth',1);
hold on;
plot(T*t,ud1,'--','LineWidth',1);
legend('u','u_T');
xlabel('Time(s)');




