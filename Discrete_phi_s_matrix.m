function [ X ] = Discrete_phi_s_matrix(s, n)
%Compute Phi*s matrix. This will be used to compute f(s) = X*w
res = size(s,2);
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


end

