load('gesture_data')

%struct = come_;
%struct = shoo_;
struct = wave_;
for i = 1:size(struct,2)
    s_struct = struct{1,i}.data(:,5);
    fx_struct = struct{1,i}.data(:,6);
    plot(s_struct, fx_struct);
    hold on
end