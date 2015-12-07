clc; clear;
%load('gesture_data_dmp_weights_basis50');
load('gesture_data_dmp_weights_rhytmic_basis40');

% Here We try my version of GMR, matlab's GMR, similarity matrix
% and we also do supervised clustering

% Discrete EM algo
demos = [static_, iu_,  triangle_, letterS_, UL_slash_, LL_slash_];

m = size(demos,2); % number of examples
n = num_basis*3; % features per demo.
W_all = zeros(n, m); % weights

for i = 1:m
    W_all(:, i) = demos{i}.weights_com';
end

W_all = featureNormalize(W_all);

K = 6; % clusters
% 
S_all = zeros(m, m);
for i = 1:m
    for j = 1:m
%        S_all(i,j) = norm(W_all(:,i) - W_all(:,j)) ;
        S_all(i,j) = (W_all(:,i)'*W_all(:,j))/(norm(W_all(:,i))*norm(W_all(:,j)));
    end
end

%figure(1)
%imagesc(S_all);
% caxis([0, 1])
% title('dotProduct Similarity Matrix of all demonstrations');
% colorbar;
% colormap('gray')

S_all_diff = zeros(m, m);
for i = 1:m
    for j = 1:m
%        S_all(i,j) = norm(demos{1,i}.weights_com - demos{1,j}.weights_com);
        S_all_diff(i,j) = norm(W_all(:,i) - W_all(:,j)) ;
%        S_all(i,j) = (W_all(:,i)'*W_all(:,j))/(norm(W_all(:,i))*norm(W_all(:,j)));
    end
end

%figure(2)
%imagesc(S_all_diff);
% caxis('auto')
% title('2-norm Diff Similarity Matrix of all demonstrations');
% colorbar;
% colormap('gray')
% colormap(flipud(colormap))




demos = {static_, iu_,  triangle_, letterS_, UL_slash_, LL_slash_};
demo_size = size(demos, 2);
weight_dim = num_basis*3;
W_mean = zeros(weight_dim, demo_size);

W_cov = zeros(num_basis*3, num_basis*3, K);
for i = 1:demo_size
    W_matrix_this_gesture = zeros(weight_dim, size(demos{i},2));
    for j = 1:size(demos{i},2)
        W_matrix_this_gesture(:,j) = demos{i}{j}.weights_com';
    end
    W_matrix_this_gesture = featureNormalize(W_matrix_this_gesture);
    W_mean(:,i) = mean(W_matrix_this_gesture, 2);
    W_cov(:,:,i) = cov(W_matrix_this_gesture');
end

pi_c = 1/K * ones(K,1);
K_cluster = {};
for i = 1:K
    K_cluster{i}.u = W_mean(:,i); %u_init(:,i);
    K_cluster{i}.pi_c = pi_c(i);
    K_cluster{i}.cov_c =  W_cov(:,:,i); % eye(num_basis*3);%
end
% 
% 
% 


% R_fit = zeros(m, K);
% for c = 1:K
%     for i = 1:m
%         resp_sum = 0;
%         for cp = 1:K
%             resp_sum = resp_sum +  K_cluster{cp}.pi_c * multivariateGaussian(W_all(:,i)', K_cluster{cp}.u, K_cluster{cp}.cov_c);
%         end    
%         R_fit(i,c) = K_cluster{c}.pi_c* multivariateGaussian(W_all(:,i)', K_cluster{c}.u, K_cluster{c}.cov_c)/resp_sum;
%     end
% end

%FEATURE SCALING IS IMPORTANT!!!!
%[X, I] = max(R_fit,[],2)


GMM_reg_performance = zeros(1, 10);
for trial = 1:10


a = -1;
b = 1;
%Start.mu = W_mean';
Start.mu = a + (b-a).*rand(6,n);

for k = 1:K
    Start.Sigma(:,:,k) = cov(W_all') + eye(n)*0.1;%W_cov(:,:,k);%
end
Start.ComponentProportion = ones(1,K)*1/K;


%rng(1);
 
%GMModels = fitgmdist(W_all',K,'RegularizationValue',0.1, 'Start', Start)
GMModels = fitgmdist(W_all',K,'RegularizationValue',0.1);
fprintf('\n GM Mean for %i Component(s)\n',j)
Mu = GMModels.mu;
Sigma = GMModels.Sigma;
ComponentProp = GMModels.ComponentProportion


% Responsibility matrix for matlab 
R_ml = zeros(m, K);
for c = 1:K
    for i = 1:m
        resp_sum = 0;
        for cp = 1:K
            resp_sum = resp_sum +  GMModels.ComponentProportion(cp) * multivariateGaussian(W_all(:,i)', Mu(cp), Sigma(:,:,cp));
        end    
        R_ml(i,c) = GMModels.ComponentProportion(c)* multivariateGaussian(W_all(:,i)', Mu(c), Sigma(:,:,c))/resp_sum;
    end
end

[X2, I2] = max(R_ml,[],2);


 m_per = m/K;
% GMM_reg_accuracy = zeros(1, K);
% for i = 1:K
%     first = (i-1)*m_per + 1;
%     last = i*m_per;
%     GMM_reg_accuracy(i) = sum(I2(first:last) == i)/m_per;
% end

GMM_reg_mistakes = 0;
for i = 1:K
    elements_in_this_cluster = sum(I2(1:m) == i);
    if elements_in_this_cluster > m_per 
        GMM_reg_mistakes = GMM_reg_mistakes + (elements_in_this_cluster - m_per);
    end
end
GMM_reg_accuracy = (m - m_per - GMM_reg_mistakes )/(m - m_per);

GMM_fit_accuracy = zeros(1, K);

% for i = 1:K
%     first = (i-1)*m_per + 1;
%     last = i*m_per;
%     GMM_fit_accuracy(i) = sum(I(first:last) == i)/m_per;
% end

GMM_fit_accuracy
GMM_reg_accuracy

GMM_reg_performance(trial) = GMM_reg_accuracy

end
GMM_reg_performance_mean = mean(GMM_reg_performance)
GMM_reg_performance_std = std(GMM_reg_performance)
%%


%%
% 
% 
% pi_c = 1/K * ones(K,1);
% K_cluster2 = {};
% a = -1;
% b = 1;
% for i = 1:K
%     K_cluster2{i}.u = a + (b-a).*rand(num_basis*3, 1) %;(:,i); %u_init(:,i);
%     K_cluster2{i}.pi_c = pi_c(i);
%     K_cluster2{i}.cov_c = cov(W_all') + eye(n)*0.1;% W_cov(:,:,i); % ;%eye(num_basis*3);%
% end

% 
% % Responsibility matrix
% R = zeros(m, K);
% 
% % Iter steps
% EMiter_total = 10;
% for EM_iter = 1:EMiter_total
%     % E-Step
%     for c = 1:K
%         for i = 1:m
%             resp_sum = 0;
%             for cp = 1:K
%                 resp_sum = resp_sum +  K_cluster2{cp}.pi_c * multivariateGaussian(W_all(:,i)', K_cluster2{cp}.u, K_cluster2{cp}.cov_c);
%             end    
%             R(i,c) = K_cluster2{c}.pi_c* multivariateGaussian(W_all(:,i)', K_cluster2{c}.u, K_cluster2{c}.cov_c)/resp_sum;
%         end
%     end
%     
%     % M-Step
%     m_c = zeros(1,K); %Responsibility assigned to cluster c
%     %Update pi_c
%     for c = 1:K
%         m_c(c) = sum(R(:,c));
%         K_cluster2{c}.pi_c = m_c(c)/m;
%     end
%     
%     %Update mu_c
%     for c = 1:K
%         data_resp_sum = 0;
%         for i = 1:m
%             data_resp_sum = data_resp_sum + R(i,c)*W_all(:,i);
%         end
%         mu_c = (1/m_c(c))*data_resp_sum;
%         K_cluster2{c}.u = mu_c;
%     end
%     % Update cov_c
%     for c = 1:K
%        cov_update_sum =  zeros(n,n);
%        for i = 1:m
%            x = W_all(:,i);
%            u = K_cluster2{c}.u;
%            cov_update_sum = cov_update_sum + R(i,c)*(x-u)*(x-u)';
%        end
%         K_cluster2{c}.cov_c = (1/m_c(c))*cov_update_sum + eye(n)*0.1;
%     end
% %     K_cluster2{1}.pi_c + K_cluster2{2}.pi_c + K_cluster2{3}.pi_c
%       EM_iter
% %      R
% end
% [X3, I3] = max(R,[],2)

