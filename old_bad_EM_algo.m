clc; clear;
load('gesture_data_dmp_weights');

% Discrete EM algo
%demos = [LL_to_UR_, UL_to_LR, LR_lowerWave, LR_upperWave]; %, wave_, shoo_, come_];
demos = [static_, iu_,  triangle_, letterS_, UL_slash_, LL_slash_];

m = size(demos,2); % number of examples
n = num_basis*3; % features per demo.
W_all = zeros(n, m); % weights

for i = 1:m
    W_all(:, i) = demos{i}.weights_com';
end

K = 4; % clusters
rng(3);


demos = {LL_to_UR_, UL_to_LR, LR_lowerWave, LR_upperWave};


demo_size = size(demos, 2);
weight_dim = num_basis*3;
W_mean = zeros(weight_dim, demo_size);
for i = 1:demo_size
    W_matrix_this_gesture = zeros(weight_dim, size(demos{i},2));
    for j = 1:size(demos{i},2)
        W_matrix_this_gesture(:,j) = demos{i}{j}.weights_com';
    end
    W_mean(:,i) = mean(W_matrix_this_gesture, 2);
end

W_all = W_all/(1e+06);

%Start.mu = W_mean';
Start.mu = [LL_to_UR_{1}.weights_com; UL_to_LR{1}.weights_com; LR_lowerWave{1}.weights_com; LR_upperWave{1}.weights_com]/(1e+06); 
for k = 1:K
    Start.Sigma(:,:,k) = cov(W_all');%eye(n)
end
Start.ComponentProportion = ones(1,K)*1/K;

%rng(3);
GMModels = fitgmdist(W_all',K,'RegularizationValue',0.1);

%GMModels = fitgmdist(W_all',K,'RegularizationValue',0.1, 'Start', Start)
fprintf('\n GM Mean for %i Component(s)\n',j)
Mu = GMModels.mu;
Sigma = GMModels.Sigma;
ComponentProp = GMModels.ComponentProportion

%  
%  
% w_min = min(min(W_all));
% w_max = max(max(W_all));
% u_init = (w_min-w_max).*rand(n,K) + w_min; %Initialize means
% save('discrete_u_init_weights.mat', 'u_init');

% Initialize EM algorithm
%load('discrete_u_init_weights.mat') % load stored mean u_init weights

u_init = [LL_to_UR_{1}.weights_com', UL_to_LR{1}.weights_com', LR_lowerWave{1}.weights_com', LR_upperWave{1}.weights_com']; 

pi_c = 1/K * ones(K,1);
K_cluster = {};
for i = 1:K
    K_cluster{i}.u = u_init(:,i);
    K_cluster{i}.pi_c = pi_c(i);
    K_cluster{i}.cov_c = cov(W_all');% 
end

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
%                 resp_sum = resp_sum +  K_cluster{cp}.pi_c * multivariateGaussian(W_all(:,i)', K_cluster{cp}.u, K_cluster{cp}.cov_c);
%             end    
%             R(i,c) = K_cluster{c}.pi_c* multivariateGaussian(W_all(:,i)', K_cluster{c}.u, K_cluster{c}.cov_c)/resp_sum;
%         end
%     end
%     
%     % M-Step
%     m_c = zeros(1,K); %Responsibility assigned to cluster c
%     %Update pi_c
%     for c = 1:K
%         m_c(c) = sum(R(:,c));
%         K_cluster{c}.pi_c = m_c(c)/m;
%     end
%     
%     %Update mu_c
%     for c = 1:K
%         data_resp_sum = 0;
%         for i = 1:m
%             data_resp_sum = data_resp_sum + R(i,c)*W_all(:,i);
%         end
%         mu_c = (1/m_c(c))*data_resp_sum;
%         K_cluster{c}.u = mu_c;
%     end
%     % Update cov_c
%     for c = 1:K
%        cov_update_sum =  zeros(n,n);
%        for i = 1:m
%            x = W_all(:,i);
%            u = K_cluster{c}.u;
%            cov_update_sum = cov_update_sum + R(i,c)*(x-u)*(x-u)';
%        end
%         K_cluster{c}.cov_c = (1/m_c(c))*cov_update_sum + eye(n)*0.1;
%     end
%     
% %     K_cluster{1}.pi_c + K_cluster{2}.pi_c + K_cluster{3}.pi_c
% %      EM_iter
% %      R
% end
% myEM_prop = m_c/m


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
