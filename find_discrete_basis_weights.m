function [ basis_weights ] = find_discrete_basis_weights( num_basis, s, func_target, plot_bool )
    % Function accepts a target function to be approximated by a sum of
    % basis functions
    % returns a vector of basis weights
    
    f_target = func_target;
    n = num_basis;
    
    %res = size(s,2);
    %X = zeros(res,n);


%     for s_j = 1:res
%         % Compute Sum of phis
%         phi_sum = 0;
%         for i = 1:n
%                center_i = i*(1/n);
%                width_i = n/center_i;   
%             phi_i = gauss_basis(width_i, center_i, s(s_j));        
%             phi_sum = phi_sum + phi_i;
%         end
%         % Store to Training Matrix X
%         for i = 1:n
%                center_i = i*(1/n);
%                width_i = n/center_i;   
%             phi_i = gauss_basis(width_i, center_i, s(s_j));        
%             X(s_j,i) = s(s_j)*phi_i/phi_sum;
%         end
%     end
    X = Discrete_phi_s_matrix(s, n);
    
    % Compute Basis Weights using a least squares fit
    w_optimal = pinv(X'*X)*(X'*f_target); % Closed form solution
    f_guess = X*w_optimal;

    if plot_bool == 1
%         figure(1)
%         for i = 1:n
%            center_i = i*(1/n);
%            width_i = n/center_i;   
%            plot(s, gauss_basis(width_i, center_i,s));
%            hold on 
%         end
        figure(2)
        hold on
        plot(s, f_target);
        plot(s, f_guess, 'ro');
        legend('target', 'BF');

        figure(3) % Plot how the basis functions look like after it has been scaled
        for i = 1:n
            hold on
            center_i = i*(1/n);
            width_i = n/center_i;  
            plot(s, gauss_basis(width_i, center_i,s)*w_optimal(i));
        end
    end
	basis_weights = w_optimal;
end
