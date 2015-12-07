clc; clear;
load('gesture_data')
Fs = 1000;            % Sampling frequency
T = 1/Fs;             % Sampling period
L = 1000;             % Length of signal
t = (0:L-1)*T;        % Time vector

S = 0.7*sin(2*pi*50*t) + 2*sin(2*pi*120*t);
%X = 0.7*sin(2*pi*50*t) + sin(2*pi*120*t);
X = S + 2*randn(size(t));

figure(1)
plot(1000*t(1:50),X(1:50))
title('Signal Corrupted with Zero-Mean Random Noise')
xlabel('t (milliseconds)')
ylabel('X(t)')

Y = fft(X);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

figure(2)
f = Fs*(0:(L/2))/L;
plot(f,P1)
title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')

Y = fft(S);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

plot(f,P1)
title('Single-Sided Amplitude Spectrum of S(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')


for i = 1:30
    figure(1)
    plot(s_wave_{i}.data(:,5), s_wave_{i}.data(:,6))
    hold on
    figure(2)
    plot(s_wave_{i}.data(:,5), s_wave_{i}.data(:,7))    
    hold on
end

% 
% for i = 1:30
%     figure(1)
%     plot(circle_{i}.data(:,5), circle_{i}.data(:,6))
%     hold on
%     figure(2)
%     plot(circle_{i}.data(:,5), circle_{i}.data(:,7))    
%     hold on
% end

