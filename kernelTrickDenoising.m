filename='train.0';
data=load(filename);
n=1000;
X=data([1:n],:);
p=size(X,2);

% Compute centered Gram matrix
sigma=0.5;
GaussianKernel=@(x,y) exp(-sum((x-y).^2)/2*sigma^2);
G=zeros(n,n);
for i=1:n
    for j=1:n
        G(i,j)=GaussianKernel(X(i,:),X(j,:));
    end 
end
%Compute eigen vectors on the centered gram matrix
U=(1/n)*ones(n,n);
G_centered=(eye(n)-U)*G*(eye(n)-U);
[V,D] = eig(G_centered);
[EigenValuesOrdered,IndexEigen]= sort(diag(D),'descend');


%d_set=[1:50:1000];
%error=zeros(size(d_set,2),1);
%for d_=1:size(d_set,2)
%d=d_set(d_); %number of eigenvectors taken in account
d=500;
FirstEigenVectors = V(:,IndexEigen(1:d));

% Computing gammas
gamma=zeros(n,1);
sigma_noise=0.3;
X_test=data(1001,:);
X_test_noised=X_test+sigma_noise*randn(1,p); %add noise 
K_temp=zeros(n,1);

for i=1:n
    K_temp(i)=GaussianKernel(X_test_noised,X(i,:))-(1/n)*sum(G(i,:),2);
end

for j=1:n
    gamma(j)=1/n+sum(K_temp.*sum(FirstEigenVectors.*repmat(FirstEigenVectors(j,:),n,1),2),1);
end


% Fixed point method
%y=rand(1,p);
y=-1+2*rand(1,p);
while (1)

z=y;

for i=1:p
    y(i)=sum(X(:,i).*exp(-sum((repmat(z,n,1)-X).^2,2)/2*sigma^2),1);    
end
y=y/sum(exp(-sum((repmat(z,n,1)-X).^2,2)/2*sigma^2),1);
if  norm(z-y) < 1e-3
    break
end
%norm(z-y)
end


%error(d_)=norm(X_test_noised-y);
subplot(2,2,1), imshow(reshape(X_test_noised,16,16)), title('Noised data')
subplot(2,2,2), imshow(reshape(y,16,16)), title('Denoised')
subplot(2,2,3), imshow(reshape(X_test,16,16)), title('Original data')
d
%end

%Gradient method (not implemented yet)

% while (1)
%  
% d_  = 1./(b-A*x(:,end));
% d  = A'*d_;            % Gradient
% H  = A'*diag(d_.^2)*A; % Hessian    
% 
% delta = H\d;
% lambda=dot(d,H*d);
% error=[error lambda];
% 
% if (lambda < 2*epsilon || compteur > max_iter-1)
%     break
% end
% 
% t_step=1;
% 
% 
% while (1)
% 
%     
%     if ( isempty(find(A*(x-t_step*delta)-b >= 0) ) ) % condition for feasible point
%             q1 = eval_barrier_objective(x-t_step*delta,A,b);
%             q2 = eval_barrier_objective(x,A,b) - alpha*t_step*dot(d,delta);
%         if (q1 < q2 || t_step*norm(delta)==0 )
%             break
%         end
%     t_step=beta*t_step;
%     end
% end
% x=x-t_step*delta;
% trajectory=[trajectory,x];
% compteur=compteur+1;
% end
