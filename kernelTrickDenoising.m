filename='train.3';
data=load(filename);
n=100;
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

%% Denoising algorithm with d=50
d=50;
FirstEigenVectors = V(:,IndexEigen(1:d));

%compute the alpha_i s
alpha = zeros(n,d);

for i=1:d
    % normalize eigenvector
    alpha(:,i) = FirstEigenVectors(:,i) / sqrt(EigenValuesOrdered(i));
    alpha(:,i) = alpha(:,i) - mean(alpha(:,i));
end

% Computing gammas
gamma=zeros(n,1);
sigma_noise=0.5;
X_test=data(500,:);
X_test_noised=X_test+(1-2*randi(1,1))*(sigma_noise*randn(1,p)); %add noise 
K_temp=zeros(n,1);

for i=1:n
    K_temp(i)=GaussianKernel(X_test_noised,X(i,:));%-(1/n)*sum(G(i,:),2);
end

for j=1:n
    gamma(j)=1/n;%+sum(K_temp.*sum(FirstEigenVectors.*repmat(FirstEigenVectors(j,:),n,1),2),1);
    coeff = 0;
    for u = 1:n
        for i = 1:d
            coeff = coeff + alpha(u,i)*alpha(j,i);
        end
        gamma(j) = gamma(j) + (K_temp(u) - mean(G(:,u))) * coeff;
    end
    
end

% Fixed point method
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

K_error=zeros(n,1);
for i=1:n
    K_error(i)=GaussianKernel(X(i,:),y);
end

subplot(2,2,1), imshow(reshape(X_test_noised,16,16),[min(X_test_noised(:)) max(X_test_noised(:))]), title('Noised data')
subplot(2,2,2), imshow(reshape(y,16,16),[min(y(:)) max(y(:))]), title('Denoised')
subplot(2,2,3), imshow(reshape(X_test,16,16),[min(X_test(:)) max(X_test(:))]), title('Original data')





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% Loop on d %%%%%%%%%%%%%%%%%%%%%

% d_set=[1:100];
% error=zeros(size(d_set,2),1);
% for d_=1:size(d_set,2)
% d=d_set(d_); %number of eigenvectors taken in account
% d=50;
% FirstEigenVectors = V(:,IndexEigen(1:d));
% 
% %compute the alpha_i s
% alpha = zeros(n,d);
% 
% for i=1:d
%     % normalize eigenvector
%     alpha(:,i) = FirstEigenVectors(:,i) / sqrt(EigenValuesOrdered(i));
%     alpha(:,i) = alpha(:,i) - mean(alpha(:,i));
% end
% 
% % Computing gammas
% gamma=zeros(n,1);
% sigma_noise=0.5;
% X_test=data(1001,:);
% X_test_noised=X_test+(1-2*randi(1,1))*(sigma_noise*randn(1,p)); %add noise 
% K_temp=zeros(n,1);
% 
% for i=1:n
%     K_temp(i)=GaussianKernel(X_test_noised,X(i,:));%-(1/n)*sum(G(i,:),2);
% end
% 
% for j=1:n
%     gamma(j)=1/n;%+sum(K_temp.*sum(FirstEigenVectors.*repmat(FirstEigenVectors(j,:),n,1),2),1);
%     coeff = 0;
%     for u = 1:n
%         for i = 1:d
%             coeff = coeff + alpha(u,i)*alpha(j,i);
%         end
%         gamma(j) = gamma(j) + (K_temp(u) - mean(G(:,u))) * coeff;
%     end
%     
% end
% 
% % Fixed point method
% y=-1+2*rand(1,p);
% while (1)
% 
% z=y;
% for i=1:p
%     y(i)=sum(X(:,i).*exp(-sum((repmat(z,n,1)-X).^2,2)/2*sigma^2),1);    
% end
% y=y/sum(exp(-sum((repmat(z,n,1)-X).^2,2)/2*sigma^2),1);
% if  norm(z-y) < 1e-3
%     break
% end
% %norm(z-y)
% end
% 
% K_error=zeros(n,1);
% for i=1:n
%     K_error(i)=GaussianKernel(X(i,:),y);
% end
% 
% error(d_)=1-2*sum(gamma.*K_error)+dot(gamma,G*gamma);
% d_
% end
% 
% plot(error)
% title('Error = f(d)')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Gradient method (not implemented yet)


