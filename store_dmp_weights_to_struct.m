function [ struct ] = store_dmp_weights_to_struct(struct, plot_bool, s_index, x_index, y_index, z_index, num_basis)
    m = size(struct,2)
    W_fx = zeros(num_basis, m);
    W_fy = zeros(num_basis, m);
    W_fz = zeros(num_basis, m);
    for i = 1:m
        s = struct{1,i}.data(:,s_index)';
        func_target_x = struct{1,i}.data(:,x_index);
        func_target_y = struct{1,i}.data(:,y_index);
        func_target_z = struct{1,i}.data(:,z_index);

        W_fx(:,i) = find_discrete_basis_weights( num_basis, s, func_target_x, plot_bool);
        W_fy(:,i) = find_discrete_basis_weights( num_basis, s, func_target_y, plot_bool);
        W_fz(:,i) = find_discrete_basis_weights( num_basis, s, func_target_z, plot_bool);    

        struct{1,i}.textdata = [struct{1,i}.textdata, 'wx', 'wy', 'wz'];
        struct{1,i}.colheaders = [struct{1,i}.colheaders, 'wx', 'wy', 'wz'];
        struct{1,i}.weights_sep = [W_fx(:,i), W_fy(:,i), W_fz(:,i)];
        struct{1,i}.weights_com = [W_fx(:,i)', W_fy(:,i)', W_fz(:,i)'];    
    end

end

