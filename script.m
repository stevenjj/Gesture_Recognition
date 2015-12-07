clc; clear;
% Test out how discrete gaussian basis functions fit


s = [0.01:0.025:1];
res = size(s,2);
center = 0.5;
width = 12.5;

figure(1)
f_target = f_t(s, 1,0.2)';
f_target2 = f_t(s, 0.5,0.4)'; % Uncomment for 2 demos
hold on
plot(s, f_target);
%plot(s, f_target2); % Uncomment for 2 demos

n_vec = [5, 10, 50]

for k = 1:size(n_vec,2)
    n = n_vec(k); % Number of basis functions

figure(k+1)
for i = 1:n
    center_i = (i)*(1/n);
    width_i = n/center_i;
    plot(s, gauss_basis(width_i, center_i,s));
    hold on 
end

w = ones(n, 1); % weights of basis functions
X = zeros(res,n);

for s_j = 1:res
    % Compute Sum of phis
    phi_sum = 0;
    for i = 1:n
        center_i = i*(1/n);
        width_i = n/center_i;  
        phi_i = gauss_basis(width_i, center_i, s(s_j));        
        phi_sum = phi_sum + phi_i;
    end
    
    % Store to Training Matrix X
    for i = 1:n
        center_i = i*(1/n);
        width_i = n/center_i;  
        phi_i = gauss_basis(width_i, center_i, s(s_j));        
        X(s_j,i) = s(s_j)*phi_i/phi_sum;
    end
    
end

w_optimal = pinv(X'*X)*(X'*f_target); % Closed form solution
%w_optimal = pinv([X;X]'*[X;X])*([X;X]'*[f_target;f_target2])  % closed form solution for 2 demonstrations

f_guess = X*w_optimal;

f_out(:,k) = f_guess;

end
figure(1)
plot(s, f_out(:,1), '+--');
plot(s, f_out(:,2), '*--');
plot(s, f_out(:,3), 'ro');
legend('target', '5BF','10BF', '50BF');
%legend('target', 'target2', '5BF','10BF', '50BF'); uncomment for 2 demos

figure(5) % Plot how the basis functions look like after it has been scaled
for i = 1:n
    hold on
    center_i = i*(1/n);
    width_i = n/center_i;  
    plot(s, gauss_basis(width_i, center_i,s)*w_optimal(i));
end



%%%%% FAILED METHOD for for finding weights
% T = zeros(res, res);
% PHI = zeros(res, n);
% for i = 1:n
%     center_i = i*(1/n);
%     width_i = n/center_i; 
%     PHI(:,i) = gauss_basis(width_i, center_i, s)';
% end
% 
% w_try = zeros(n, 1);
% for i = 1:n
%     for s_j = 1:res
%         T(s_j, s_j) = PHI(s_j, i);
%     end
%     w_try(i,1) = (s*T*f_target)/(s*T*s');
% end
