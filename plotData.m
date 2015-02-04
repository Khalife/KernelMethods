%filename='train.0';
data= [load('train.0'); load('train.1'); load('train.2'); load('train.3')];
label = [0 * ones(size(load('train.0'),1),1); 1 * ones(size(load('train.1'),1),1); 2 * ones(size(load('train.2'),1),1);3 * ones(size(load('train.3'),1),1)];
n=300;
permut = randperm(size(data,1));
permut = permut(1:n);

X=data(permut,:);
label = label(permut);
p=size(X,2);

% Compute centered Gram matrix for gaussian kernel
sigma=0.5;
linKernel=@(x,y) x*y';
G=zeros(n,n);
for i=1:n
    for j=1:n
        G(i,j)=linKernel(X(i,:),X(j,:));
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

%plot data in 2d using the projection on the 2 first dimensions
figure(1); hold on;
dim1 = G*alpha(:,1);
dim2 = G*alpha(:,2);
if sum(label == 0) > 0
    scatter(dim1(label == 0), dim2(label == 0), 'full', 'b');
end
if sum(label == 1) > 0
    scatter(dim1(label == 1), dim2(label == 1), 'full', 'r');
end
if sum(label == 2) > 0
    scatter(dim1(label == 2), dim2(label == 2), 'full', 'g');
end
if sum(label == 3) > 0
    scatter(dim1(label == 3), dim2(label == 3), 'full', 'k');
end



% Compute centered Gram matrix for gaussian kernel
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

%plot data in 2d using the projection on the 2 first dimensions
figure(2);hold on;
dim1 = G*alpha(:,1);
dim2 = G*alpha(:,2);
if sum(label == 0) > 0
    scatter(dim1(label == 0), dim2(label == 0), 'full', 'b');
end
if sum(label == 1) > 0
    scatter(dim1(label == 1), dim2(label == 1), 'full', 'r');
end
if sum(label == 2) > 0
    scatter(dim1(label == 2), dim2(label == 2), 'full', 'g');
end
if sum(label == 3) > 0
    scatter(dim1(label == 3), dim2(label == 3), 'full', 'k');
end

% Compute centered Gram matrix for gaussian kernel
sigma=1;
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

%plot data in 2d using the projection on the 2 first dimensions
figure(4);hold on;
dim1 = G*alpha(:,1);
dim2 = G*alpha(:,2);
if sum(label == 0) > 0
    scatter(dim1(label == 0), dim2(label == 0), 'full', 'b');
end
if sum(label == 1) > 0
    scatter(dim1(label == 1), dim2(label == 1), 'full', 'r');
end
if sum(label == 2) > 0
    scatter(dim1(label == 2), dim2(label == 2), 'full', 'g');
end
if sum(label == 3) > 0
    scatter(dim1(label == 3), dim2(label == 3), 'full', 'k');
end


% Compute centered Gram matrix for gaussian kernel
sigma=0.5;
polyKernel=@(x,y) (x*y' + 1)^2;
G=zeros(n,n);
for i=1:n
    for j=1:n
        G(i,j)=polyKernel(X(i,:),X(j,:));
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

%plot data in 2d using the projection on the 2 first dimensions
figure(3);hold on;
dim1 = G*alpha(:,1);
dim2 = G*alpha(:,2);
if sum(label == 0) > 0
    scatter(dim1(label == 0), dim2(label == 0), 'full', 'b');
end
if sum(label == 1) > 0
    scatter(dim1(label == 1), dim2(label == 1), 'full', 'r');
end
if sum(label == 2) > 0
    scatter(dim1(label == 2), dim2(label == 2), 'full', 'g');
end
if sum(label == 3) > 0
    scatter(dim1(label == 3), dim2(label == 3), 'full', 'k');
end