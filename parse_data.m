clc; clear;

%% Parse Data from txt file
%Lower Left to Upper Right Straight Swipe
n = 12; % Number of Data
for i=1:n 
    LL_to_UR_{i} = importdata(sprintf('training_data/LL_to_UR%i.txt', i));
end

%Upper Left to Lower Right Swipe
n = 11; % Number of Data
for i=1:n 
    UL_to_LR{i} = importdata(sprintf('training_data/UL_to_LR%i.txt', i));
end

%Left to Right lower wave
n = 13; % Number of Data
for i=1:n 
    LR_lowerWave{i} = importdata(sprintf('training_data/LR_lowerWave%i.txt', i));
end

%Left to Right upper wave
n = 11; % Number of Data
for i=1:n 
    LR_upperWave{i} = importdata(sprintf('training_data/LR_upperWave%i.txt', i));
end

% Wave gesture
n = 10; % Number of Data
for i=1:n 
    wave_{i} = importdata(sprintf('training_data/wave_%i.txt', i));
end

% Come forth gesture
n = 10; % Number of Data
for i=1:n 
    come_{i} = importdata(sprintf('training_data/come_%i.txt', i));
end

% Shoo gesture
n = 10; % Number of Data
for i=1:n 
    shoo_{i} = importdata(sprintf('training_data/shoo_%i.txt', i));
end


% inverted U
n = 30; % Number of Data
for i=1:n 
    iu_{i} = importdata(sprintf('training_data/iu_%i.txt', i));
end

% letterS
n = 30; % Number of Data
for i=1:n 
    letterS_{i} = importdata(sprintf('training_data/s_%i.txt', i));
end

% static
n = 30; % Number of Data
for i=1:n 
    static_{i} = importdata(sprintf('training_data/static_%i.txt', i));
end

% triangle
n = 30; % Number of Data
for i=1:n 
    triangle_{i} = importdata(sprintf('training_data/triangle_%i.txt', i));
end

% LL_slash
n = 30; % Number of Data
for i=1:n 
    LL_slash_{i} = importdata(sprintf('training_data/LL_slash_%i.txt', i));
end

% UL_slash
n = 30; % Number of Data
for i=1:n 
    UL_slash_{i} = importdata(sprintf('training_data/UL_slash_%i.txt', i));
end

% iu_spatial
n = 5; % Number of Data
for i=1:n 
    iu_spatial_{i} = importdata(sprintf('training_data/iu_3%ispatial.txt', i));
end


% triangle_spatial
n = 5; % Number of Data
for i=1:n 
    triangle_spatial_{i} = importdata(sprintf('training_data/triangle_3%ispatial.txt', i));
end

% letterS_spatial
n = 5; % Number of Data
for i=1:n 
    letterS_spatial_{i} = importdata(sprintf('training_data/s_3%ispatial.txt', i));
end

% s_wave
n = 30; % Number of Data
for i=1:n 
    s_wave_{i} = importdata(sprintf('training_data/s_wave_%i.txt', i));
end

% circle_
n = 30; % Number of Data
for i=1:n 
    circle_{i} = importdata(sprintf('training_data/circle_%i.txt', i));
end


%%
% Move all points to start at (0,0,0)

for i = 1:size(LL_to_UR_,2)    
    xo = LL_to_UR_{1,i}.data(1,2); %starting xo position
    yo = LL_to_UR_{1,i}.data(1,3); %starting yo position
    zo = LL_to_UR_{1,i}.data(1,4); %starting zo position 
    
    x = LL_to_UR_{1,i}.data(:,2);
    y = LL_to_UR_{1,i}.data(:,3);
    z = LL_to_UR_{1,i}.data(:,4);    
    
    % Update x,y,z positions
    LL_to_UR_{1,i}.data(:,2) = x - xo;
    LL_to_UR_{1,i}.data(:,3) = y - yo;
    LL_to_UR_{1,i}.data(:,4) = z - zo;    
    
    x1 = LL_to_UR_{1,i}.data(:,2);
    y1 = LL_to_UR_{1,i}.data(:,3);
    z1 = LL_to_UR_{1,i}.data(:,4);   
    % hold on
    % plot3(x1,y1,z1, 's--')
end

for i = 1:size(UL_to_LR,2) 
    xo = UL_to_LR{1,i}.data(1,2); %starting xo position
    yo = UL_to_LR{1,i}.data(1,3); %starting yo position
    zo = UL_to_LR{1,i}.data(1,4); %starting zo position 

    x = UL_to_LR{1,i}.data(:,2);
    y = UL_to_LR{1,i}.data(:,3);
    z = UL_to_LR{1,i}.data(:,4);    

    % Update x,y,z positions
    UL_to_LR{1,i}.data(:,2) = x - xo;
    UL_to_LR{1,i}.data(:,3) = y - yo;
    UL_to_LR{1,i}.data(:,4) = z - zo;    
    
    x1 = UL_to_LR{1,i}.data(:,2);
    y1 = UL_to_LR{1,i}.data(:,3);
    z1 = UL_to_LR{1,i}.data(:,4);   
%     hold on
%     plot3(x1,y1,z1, 's--')
end


for i = 1:size(LR_lowerWave,2)    
    xo = LR_lowerWave{1,i}.data(1,2); %starting xo position
    yo = LR_lowerWave{1,i}.data(1,3); %starting yo position
    zo = LR_lowerWave{1,i}.data(1,4); %starting zo position 
    
    x = LR_lowerWave{1,i}.data(:,2);
    y = LR_lowerWave{1,i}.data(:,3);
    z = LR_lowerWave{1,i}.data(:,4);  
    
    % Update x,y,z positions
    LR_lowerWave{1,i}.data(:,2) = x - xo;
    LR_lowerWave{1,i}.data(:,3) = y - yo;
    LR_lowerWave{1,i}.data(:,4) = z - zo;    
    
    x1 = LR_lowerWave{1,i}.data(:,2);
    y1 = LR_lowerWave{1,i}.data(:,3);
    z1 = LR_lowerWave{1,i}.data(:,4);   
%     hold on
%     plot3(x1,y1,z1, 's--')
%     i
%     pause(3)
end


for i = 1:size(LR_upperWave,2)    
    xo = LR_upperWave{1,i}.data(1,2); %starting xo position
    yo = LR_upperWave{1,i}.data(1,3); %starting yo position
    zo = LR_upperWave{1,i}.data(1,4); %starting zo position 
    
    x = LR_upperWave{1,i}.data(:,2);
    y = LR_upperWave{1,i}.data(:,3);
    z = LR_upperWave{1,i}.data(:,4);  
    
    % Update x,y,z positions
    LR_upperWave{1,i}.data(:,2) = x - xo;
    LR_upperWave{1,i}.data(:,3) = y - yo;
    LR_upperWave{1,i}.data(:,4) = z - zo;    
    
    x1 = LR_upperWave{1,i}.data(:,2);
    y1 = LR_upperWave{1,i}.data(:,3);
    z1 = LR_upperWave{1,i}.data(:,4);   
%     hold on
%     plot3(x1,y1,z1, 's--')
%     i
%     pause(3)
end

for i = 1:size(wave_,2)    
    xo = wave_{1,i}.data(1,2); %starting xo position
    yo = wave_{1,i}.data(1,3); %starting yo position
    zo = wave_{1,i}.data(1,4); %starting zo position 
    
    x = wave_{1,i}.data(:,2);
    y = wave_{1,i}.data(:,3);
    z = wave_{1,i}.data(:,4);  
    
    % Update x,y,z positions
    wave_{1,i}.data(:,2) = x - xo;
    wave_{1,i}.data(:,3) = y - yo;
    wave_{1,i}.data(:,4) = z - zo;    
    
    x1 = wave_{1,i}.data(:,2);
    y1 = wave_{1,i}.data(:,3);
    z1 = wave_{1,i}.data(:,4);   
    hold on
    plot3(x1,y1,z1, 's--')
%     i
%     pause(3)
end


for i = 1:size(come_,2)    
    xo = come_{1,i}.data(1,2); %starting xo position
    yo = come_{1,i}.data(1,3); %starting yo position
    zo = come_{1,i}.data(1,4); %starting zo position 
    
    x = come_{1,i}.data(:,2);
    y = come_{1,i}.data(:,3);
    z = come_{1,i}.data(:,4);  
    
    % Update x,y,z positions
    come_{1,i}.data(:,2) = x - xo;
    come_{1,i}.data(:,3) = y - yo;
    come_{1,i}.data(:,4) = z - zo;    
    
    x1 = come_{1,i}.data(:,2);
    y1 = come_{1,i}.data(:,3);
    z1 = come_{1,i}.data(:,4);   
%     hold on
%     plot3(x1,y1,z1, 's--')
%     i
%     pause(3)
end


for i = 1:size(shoo_,2)    
    xo = shoo_{1,i}.data(1,2); %starting xo position
    yo = shoo_{1,i}.data(1,3); %starting yo position
    zo = shoo_{1,i}.data(1,4); %starting zo position 
    
    x = shoo_{1,i}.data(:,2);
    y = shoo_{1,i}.data(:,3);
    z = shoo_{1,i}.data(:,4);  
    
    % Update x,y,z positions
    shoo_{1,i}.data(:,2) = x - xo;
    shoo_{1,i}.data(:,3) = y - yo;
    shoo_{1,i}.data(:,4) = z - zo;    
    
    x1 = shoo_{1,i}.data(:,2);
    y1 = shoo_{1,i}.data(:,3);
    z1 = shoo_{1,i}.data(:,4);   
%     hold on
%     plot3(x1,y1,z1, 's--')
%     i
%     pause(3)
end
save('gesture_data')