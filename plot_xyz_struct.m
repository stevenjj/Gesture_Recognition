function [ output_args ] = plot_xyz_struct( struct )
%     tf::Vector3 dmp_to_pr2_frame(tf::Vector3 vector_pos){
%     return tf::Vector3(-vector_pos.getZ(), vector_pos.getX(), -vector_pos.getY() );
% }

for i = 1:size(struct,2)
    hold on
    t = struct{1,i}.data(:,1);
    x = struct{1,i}.data(:,2);
    y = struct{1,i}.data(:,3);
    z = struct{1,i}.data(:,4);
    plot3(x,y,z, 's--')
    xlabel('x')
    ylabel('y')
    zlabel('z')

end

end

