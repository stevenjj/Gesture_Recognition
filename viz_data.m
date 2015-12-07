clc; clear;

load('gesture_data');
% 
% figure(1)
% plot_xyz_struct(LL_to_UR_)
% 
% figure(2)
% plot_xyz_struct(UL_to_LR)
% 
% figure(3)
% plot_xyz_struct(LR_lowerWave)
% 
% figure(4)
% plot_xyz_struct(LR_upperWave)
% 
% figure(5)
% plot_xyz_struct(wave_)
% 
% figure(6)
% plot_xyz_struct(come_)
% 
% figure(7)
% plot_xyz_struct(shoo_)
% 

figure(8)
plot_xyz_struct(static_)

figure(9)
plot_xyz_struct(iu_)

figure(10)
plot_xyz_struct(triangle_)

figure(11)
plot_xyz_struct(letterS_)

figure(12)
plot_xyz_struct(UL_slash_)

figure(13)
plot_xyz_struct(LL_slash_)

figure(14)
% hold on 
% plot_xyz_struct(iu_)

plot_xyz_struct(iu_spatial_)

figure(15)
% hold on
% plot_xyz_struct(triangle_)
plot_xyz_struct(triangle_spatial_)

figure(16)
% hold on
% plot_xyz_struct(letterS_)
plot_xyz_struct(letterS_spatial_)

figure(17)
plot_xyz_struct(s_wave_)

figure(18)
plot_xyz_struct(circle_)


figure(19)
hold on
subplot(4,2,1)
plot_xyz_struct(static_)
subplot(4,2,2)
plot_xyz_struct(iu_)
subplot(4,2,3)
plot_xyz_struct(triangle_)
subplot(4,2,4)
plot_xyz_struct(letterS_)
axis('auto')
subplot(4,2,5)
plot_xyz_struct(UL_slash_)
subplot(4,2,6)
plot_xyz_struct(LL_slash_)
subplot(4,2,7)
plot_xyz_struct(s_wave_)
subplot(4,2,8)
plot_xyz_struct(circle_)



% figure(2)
% for i = 1:size(LL_to_UR_,2)
%     hold on
%     t = LL_to_UR_{1,i}.data(:,1);
%     num = size(t,1);
%     x = zeros(num, 1);
%     y = zeros(num, 1);
%     z = zeros(num, 1);    
%     for j = 1:num
%         x(j) = LL_to_UR_{1,i}.data(j,2);
%         y(j) = LL_to_UR_{1,i}.data(j,3);
%         z(j) = LL_to_UR_{1,i}.data(j,4);    
%         plot3(x,y,z, 's--')
%         pause(0.5);
%     end
% 
% end