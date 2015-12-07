clc; clear;
load('gesture_data')

plot_bool = 0;
s_index = 5;
x_index = 6;
y_index = 7;
z_index = 8;
num_basis = 5;

% LL_to_UR_ = store_dmp_weights_to_struct(LL_to_UR_, plot_bool, s_index, x_index, y_index, z_index, num_basis);
% LR_lowerWave = store_dmp_weights_to_struct(LR_lowerWave, plot_bool, s_index, x_index, y_index, z_index, num_basis);
% LR_upperWave = store_dmp_weights_to_struct(LR_upperWave, plot_bool, s_index, x_index, y_index, z_index, num_basis);
% UL_to_LR = store_dmp_weights_to_struct(UL_to_LR, plot_bool, s_index, x_index, y_index, z_index, num_basis);
% wave_ = store_dmp_weights_to_struct(wave_, plot_bool, s_index, x_index, y_index, z_index, num_basis);
% shoo_ = store_dmp_weights_to_struct(shoo_, plot_bool, s_index, x_index, y_index, z_index, num_basis);
% come_ = store_dmp_weights_to_struct(come_, plot_bool, s_index, x_index, y_index, z_index, num_basis);
% 

disp('Fitting Inverted U...')
iu_ = store_dmp_weights_to_struct(iu_, plot_bool, s_index, x_index, y_index, z_index, num_basis);
disp('DONE!')

disp('Fitting Letter S...')
letterS_ = store_dmp_weights_to_struct(letterS_, plot_bool, s_index, x_index, y_index, z_index, num_basis);
disp('DONE!')

disp('Fitting static...')
static_ = store_dmp_weights_to_struct(static_, plot_bool, s_index, x_index, y_index, z_index, num_basis);
disp('DONE!')


disp('Fitting Triangle...')
triangle_ = store_dmp_weights_to_struct(triangle_, plot_bool, s_index, x_index, y_index, z_index, num_basis);
disp('DONE!')

disp('Fitting UL slash...')
UL_slash_ = store_dmp_weights_to_struct(UL_slash_, plot_bool, s_index, x_index, y_index, z_index, num_basis);
disp('DONE!')

disp('Fitting LL slash...')
LL_slash_ = store_dmp_weights_to_struct(LL_slash_, plot_bool, s_index, x_index, y_index, z_index, num_basis);
disp('DONE!')
% 

disp('Fitting iu spatial...')
iu_spatial_ = store_dmp_weights_to_struct(iu_spatial_, plot_bool, s_index, x_index, y_index, z_index, num_basis);
disp('DONE!')

disp('Fitting letterS spatial...')
letterS_spatial_ = store_dmp_weights_to_struct(letterS_spatial_, plot_bool, s_index, x_index, y_index, z_index, num_basis);
disp('DONE!')

disp('Fitting triangle spatial...')
triangle_spatial_ = store_dmp_weights_to_struct(triangle_spatial_, plot_bool, s_index, x_index, y_index, z_index, num_basis);
disp('DONE!')

disp('Fitting slow wave ')
s_wave_ = store_dmp_weights_to_struct(s_wave_, plot_bool, s_index, x_index, y_index, z_index, num_basis);
disp('DONE!')

disp('Fitting circle...')
circle_ = store_dmp_weights_to_struct(circle_, plot_bool, s_index, x_index, y_index, z_index, num_basis);
disp('DONE!')


%load('gesture_data_dmp_weights')

demos = [static_, iu_, letterS_, LL_to_UR_, LR_lowerWave, LR_upperWave, UL_to_LR, wave_, shoo_, come_, circle_];


demos_new = [static_, iu_,  triangle_, letterS_, UL_slash_, LL_slash_, s_wave_, circle_];
%demos_new = [letterS_, static_, iu_,  triangle_, UL_slash_, LL_slash_, LR_lowerWave, LR_upperWave, UL_to_LR, wave_, shoo_, come_];

m_weights = 0;
for i = 1:size(demos_new,2)
    m_weights = m_weights + size(demos_new(i),2);
end

%m_d1 = size(LL_to_UR_, 2);
S_all = zeros(m_weights, m_weights);
for i = 1:m_weights
    for j = 1:m_weights
        %S_all(i,j) = norm(demos_new{1,i}.weights_com - demos_new{1,j}.weights_com);        
       S_all(i,j) = (demos_new{1,i}.weights_com)*(demos_new{1,j}.weights_com')/(norm(demos_new{1,i}.weights_com)*norm(demos_new{1,j}.weights_com));    
    end
end

figure(10)
imagesc(S_all);
colorbar;
caxis([0,1])
%caxis('auto')

%save('gesture_data_dmp_weights_rhytmic_basis35')

