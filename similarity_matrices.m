clc; clear;
% Load the gesture data with the DMP weights already fitted and placed into
% the gesture_data structure.
load('gesture_data_dmp_weights');
%demos = [LL_to_UR_, UL_to_LR, LR_lowerWave, LR_upperWave, wave_, shoo_, come_];
demos = [static_, iu_,  triangle_, letterS_, UL_slash_, LL_slash_];
m_weights = 0;

for i = 1:size(demos,2)
    m_weights = m_weights + size(demos(i),2);
end

%m_d1 = size(LL_to_UR_, 2);
S_all = zeros(m_weights, m_weights);

for i = 1:m_weights
    for j = 1:m_weights
%        S_all(i,j) = norm(demos{1,i}.weights_com - demos{1,j}.weights_com);
        S_all(i,j) = (demos{1,i}.weights_com)*(demos{1,j}.weights_com')/(norm(demos{1,i}.weights_com)*norm(demos{1,j}.weights_com));
    end
end

figure(1)
imagesc(S_all);
caxis([0, 1])
title('Similarity Matrix of all demonstrations');
colorbar;


% Calculate the mean of the weights
%demos = {LL_to_UR_, UL_to_LR, LR_lowerWave, LR_upperWave, wave_, shoo_, come_};
demos = {static_, iu_,  triangle_, letterS_, UL_slash_, LL_slash_};
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

% Calculate Similarity Matrix 
S_mean = zeros(size(W_mean,2), size(W_mean,2));
for i = 1:size(W_mean,2)
    for j = 1:size(W_mean,2)
%        S_mean(i,j) = norm(W_mean(:,i) - W_mean(:,j));
        S_mean(i,j) = W_mean(:,i)'*W_mean(:,j)/(norm(W_mean(:,i))*norm(W_mean(:,j)));        
    end
end

figure(2)
imagesc(S_mean)
title('Similarity matrix of Demonstration Means')
caxis([0.3, 1])
colorbar

% Calculate the mean of the weights of only the discrete motions
%demos = {LL_to_UR_, UL_to_LR, LR_lowerWave, LR_upperWave};
demos = {static_, iu_,  triangle_, letterS_, UL_slash_, LL_slash_};
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

% Calculate Similarity Matrix 
S_mean_discrete = zeros(size(W_mean,2), size(W_mean,2));
for i = 1:size(W_mean,2)
    for j = 1:size(W_mean,2)
%        S_mean_discrete(i,j) = norm(W_mean(:,i) - W_mean(:,j));
        S_mean_discrete(i,j) = W_mean(:,i)'*W_mean(:,j)/(norm(W_mean(:,i))*norm(W_mean(:,j)));
%        S_mean(i,j) = W_mean(:,i)'*W_mean(:,j)/(norm(W_mean(:,i))*norm(W_mean(:,j)));  
    end
end

figure(3)
imagesc(S_mean_discrete)
title('Similarity matrix of Discrete Demonstration Means')
caxis([0, 1])
colorbar

% Gather all the demonstrations into one matrix
%demos_all_struct = {LL_to_UR_, UL_to_LR, LR_lowerWave, LR_upperWave, wave_, shoo_, come_};
%demos_all_weights = [LL_to_UR_, UL_to_LR, LR_lowerWave, LR_upperWave, wave_, shoo_, come_];

demos_all_struct = {static_, iu_,  triangle_, letterS_, UL_slash_, LL_slash_};
demos_all_weights = [static_, iu_,  triangle_, letterS_, UL_slash_, LL_slash_];

demo_all_weights_size = size(demos_all_weights, 2);
weight_dim = num_basis*3;
W_all_weights = zeros(demo_all_weights_size, weight_dim);
for i = 1:demo_all_weights_size    
    W_all_weights(i,:) = demos_all_weights{1,i}.weights_com;
end


% Failed attempt
% ** Do Principal Comoponent Analysis
% Calculate  covariance matrix
sigma = (1/demo_all_weights_size) * W_all_weights' * W_all_weights;
% Do singular value decomposition
[U,S,V] = svd(sigma);
k = 3 % two dimensions
Ureduce = U(:,1:k);      % take the first k directions

Z = W_all_weights*Ureduce;
%z = Ureduce' * x;        % compute the projected data points
%plot(Z(:,1), Z(:,2), 's');
%plot3(Z(:,1), Z(:,2),Z(:,3), 's');

figure(4)
counter = 1;
for i = 1:size(demos_all_struct,2)
    this_gesture_size = size(demos_all_struct{i},2);
    Z = W_all_weights(counter:counter+(this_gesture_size-1),:) * Ureduce;
    W = W_all_weights(counter:counter+(this_gesture_size-1),:);
%    plot(W(:,1),W(:,2),'s');
 %   plot(Z(:,1), Z(:,2), 's');
    plot3(Z(:,1), Z(:,2),Z(:,3), 's');
    hold on
    counter = counter+this_gesture_size;
end
% 
