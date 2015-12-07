clc; clear;
load('gesture_data_dmp_weights');

% Discrete EM algo
demos = [static_, iu_,  triangle_, letterS_, UL_slash_, LL_slash_];
K = 6;
m = size(demos,2); % number of examples
m_all = m/K;
train_ratio = 2/3;
m_train = m/K*train_ratio;
n = num_basis*3; % features per demo.
W_all = zeros(n, m); % weights

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


S_train = zeros(m_total_training_data, m_total_training_data);
for i = 1:m_total_training_data
    for j = 1:m_total_training_data
%        S_all(i,j) = norm(W_all(:,i) - W_all(:,j)) ;
        S_train(i,j) = (W_train(:,i)'*W_train(:,j))/(norm(W_train(:,i))*norm(W_train(:,j)));
    end
end
figure(1)
imagesc(S_train);
caxis([0, 1])
title('Similarity Matrix of Training demonstrations');
colorbar;
colormap('parula')


R_fit = zeros(m, K);
for c = 1:K
    for i = 1:m
        resp_sum = 0;
        for cp = 1:K
            resp_sum = resp_sum +  K_cluster{cp}.pi_c * multivariateGaussian(W_all(:,i)', K_cluster{cp}.u, K_cluster{cp}.cov_c);
        end    
        R_fit(i,c) = K_cluster{c}.pi_c* multivariateGaussian(W_all(:,i)', K_cluster{c}.u, K_cluster{c}.cov_c)/K_cluster{c}.pi_c* multivariateGaussian(W_all(:,i)', K_cluster{c}.u, K_cluster{c}.cov_c);        
   end
end

[X, I] = max(R_fit,[],2)
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
