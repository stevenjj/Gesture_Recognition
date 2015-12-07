clc; clear;
%load('gesture_data_dmp_weights2');
load('gesture_data_dmp_weights_basis25');

% Discrete EM algo
demos = [static_, iu_,  triangle_, letterS_, UL_slash_, LL_slash_];
K = 6;
m = size(demos,2); % number of examples
m_all = m/K;
train_ratio = 2/3;
m_train = m/K*train_ratio;
m_cv = floor(m/K*(1-train_ratio));

n = num_basis*3; % features per demo.
W_all = zeros(n, m); % weights

GMFit_accuracy = zeros(1, 10);
GMSpatial_accuracy = zeros(1, 10);
GMMDiscrete_cv_accuracy = zeros(1, 10);

for trial = 1:10

for i = 1:m
    W_all(:, i) = demos{i}.weights_com';
end
W_all = featureNormalize(W_all);


% Create K Gaussian Mixtures
pi_c = 1/K * ones(K,1);
K_cluster = {};
for i = 1:K
    K_cluster{i}.weights_all = W_all(:,m_all*(i-1)+1:m_all*i);
    %Permute weights here %%%%
    K_cluster{i}.weights_all = K_cluster{i}.weights_all(:,randperm(m_all)); 
    K_cluster{i}.weights_train = K_cluster{i}.weights_all(:,1:m_train);   
    K_cluster{i}.weights_test = K_cluster{i}.weights_all(:,m_train+1:m_all);
    
    K_cluster{i}.u = mean(K_cluster{i}.weights_train,2); 
    K_cluster{i}.pi_c = pi_c(i);    
    K_cluster{i}.cov_c =  cov(K_cluster{i}.weights_train') + eye(n)*0.0001; %Regularization term
end

m_total_training_data = m*train_ratio;
W_train = zeros(n, m_total_training_data);
for i = 1:K
     W_train(:,m_train*(i-1)+1:m_train*i) = K_cluster{i}.weights_train;
end

m_total_cv_test_data = floor(m*(1-train_ratio));
W_cv = zeros(n, m_total_cv_test_data);
for i = 1:K
     W_cv(:,m_cv*(i-1)+1:m_cv*i) = K_cluster{i}.weights_test;
end


S_train = zeros(m_total_training_data, m_total_training_data);
for i = 1:m_total_training_data
    for j = 1:m_total_training_data
%        S_all(i,j) = norm(W_all(:,i) - W_all(:,j)) ;
        S_train(i,j) = (W_train(:,i)'*W_train(:,j))/(norm(W_train(:,i))*norm(W_train(:,j)));
    end
end
% figure(1)
% imagesc(S_train);
% caxis([0, 1])
% title('Similarity Matrix of Training demonstrations');
% colorbar;
% colormap('parula')

%% Test across all training dataset

R_fit = zeros(m, K);
for c = 1:K
    for i = 1:m
        resp_sum = 0;
        for cp = 1:K
            resp_sum = resp_sum +  K_cluster{cp}.pi_c * multivariateGaussian(W_all(:,i)', K_cluster{cp}.u, K_cluster{cp}.cov_c);
        end    
        R_fit(i,c) = K_cluster{c}.pi_c* multivariateGaussian(W_all(:,i)', K_cluster{c}.u, K_cluster{c}.cov_c)/resp_sum;        
   end
end
[X, I] = max(R_fit,[],2);
GMM_fit_accuracy = zeros(1, K);
GMM_total_correct = 0;
m_per = m/K;
for i = 1:K
    first = (i-1)*m_per + 1;
    last = i*m_per;
    GMM_fit_accuracy(i) = sum(I(first:last) == i)/m_per;
    GMM_total_correct = GMM_total_correct + sum(I(first:last) == i);
end
GMM_fit_accuracy
GMM_total_accuracy = GMM_total_correct/m



%%
%%% Spatial demonstration tests


%expected clusters 2, 3, 4 respectively.
spatial_demos = [iu_spatial_, triangle_spatial_, letterS_spatial_];
m_spatial = size(spatial_demos,2); 
W_spatial = zeros(n, m_spatial); % weights
for i = 1:m_spatial
    W_spatial(:, i) = spatial_demos{i}.weights_com';
end
W_spatial = featureNormalize(W_spatial);

S_spatial = zeros(m_spatial, m_spatial);
for i = 1:m_spatial
    for j = 1:m_spatial
        S_spatial(i,j) = (W_spatial(:,i)'*W_spatial(:,j))/(norm(W_spatial(:,i))*norm(W_spatial(:,j)));
    end
end
% 
% figure(2)
% imagesc(S_spatial);
% caxis([0, 1])
% title('Similarity Matrix of Spatial demonstrations');
% colorbar;
% colormap('parula')

R_spatial = zeros(m_spatial, K);
for c = 1:K
    for i = 1:m_spatial
        resp_sum = 0;
        for cp = 1:K
            resp_sum = resp_sum +  K_cluster{cp}.pi_c * multivariateGaussian(W_spatial(:,i)', K_cluster{cp}.u, K_cluster{cp}.cov_c);
        end    
        R_spatial(i,c) = K_cluster{c}.pi_c* multivariateGaussian(W_spatial(:,i)', K_cluster{c}.u, K_cluster{c}.cov_c)/resp_sum;        
   end
end

[Xspatial, Ispatial] = max(R_spatial,[],2);

GMM_spatial_fit_accuracy = zeros(1, K);
GMM_spatial_total_correct = 0;
m_per = m_spatial/3;
for i = 2:4
    first = (i-2)*m_per + 1;
    last = (i-1)*m_per;
    GMM_spatial_fit_accuracy(i) = sum(Ispatial(first:last) == i)/m_per;
    GMM_spatial_total_correct = GMM_spatial_total_correct + sum(Ispatial(first:last) == i);
end
GMM_spatial_fit_accuracy
GMM_spatial_total_accuracy = GMM_spatial_total_correct/(m_spatial)


%% Test using only cross validation dataset

R_cv = zeros(m_total_cv_test_data, K);
for c = 1:K
    for i = 1:m_total_cv_test_data
        resp_sum = 0;
        for cp = 1:K
            resp_sum = resp_sum +  K_cluster{cp}.pi_c * multivariateGaussian(W_cv(:,i)', K_cluster{cp}.u, K_cluster{cp}.cov_c);
        end    
        R_cv(i,c) = K_cluster{c}.pi_c* multivariateGaussian(W_cv(:,i)', K_cluster{c}.u, K_cluster{c}.cov_c)/resp_sum;        
   end
end

[Xcv, Icv] = max(R_cv,[],2);
GMM_cv_accuracy = zeros(1, K);
GMM_cv_total_correct = 0;
m_cv_per = m_total_cv_test_data/K;
for i = 1:K
    first = (i-1)*m_cv_per + 1;
    last = i*m_cv_per;
    GMM_cv_accuracy(i) = sum(Icv(first:last) == i)/m_cv_per;
    GMM_cv_total_correct = GMM_cv_total_correct + sum(Icv(first:last) == i);
end
GMM_cv_accuracy;
GMM_cv_total_accuracy = GMM_cv_total_correct/m_total_cv_test_data;


GMFit_accuracy(trial) = GMM_total_accuracy
GMSpatial_accuracy(trial) = GMM_spatial_total_accuracy
GMMDiscrete_cv_accuracy(trial) = GMM_cv_total_accuracy
end
% GMFit_accuracy
% GMSpatial_accuracy

GMFit_accuracy_mean = mean(GMFit_accuracy)
GMFit_accuracy_std = std(GMFit_accuracy)

GMSpatial_accuracy_mean = mean(GMSpatial_accuracy)
GMSpatial_accuracy_std = std(GMSpatial_accuracy)

GMMDiscrete_cv_accuracy_mean = mean(GMMDiscrete_cv_accuracy)
GMMDiscrete_cv_accuracy_std = std(GMMDiscrete_cv_accuracy)