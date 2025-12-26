[speech, Fs] = audioread('/MATLAB Drive/speech.wav');
if size(speech, 2) > 1
    speech = speech(:, 1); % Use the left channel if stereo
end
% speech = speech(1:15*Fs);

pre_emphasis_coeff = 0.9;
speech = filter([1, -pre_emphasis_coeff], 1, speech);

% Parameters
segment_time = 0.020; 
order = 100;
SNR_dB = 20;

% Add noise
speech = add_noise(speech, SNR_dB);

% block parameters
segment_length = round(segment_time * Fs);
step = segment_length / 2; 
num_segments = floor((length(speech) - segment_length) / step) + 1;

% Initialize outputs
synthesized = zeros(length(speech), 1);
synthesized_noisy = zeros(length(speech), 1);

% Process each segment
for i = 1:num_segments
    start = (i-1) * step + 1;
    endd = start + segment_length - 1;
    if endd > length(speech)
        break;
    end
    x = speech(start:endd);
    
    % Windowing
    win = hamming(segment_length);
    x_windowed = x .* win;
    
    % LPC coefficients
    a = my_lpc(x_windowed, order);
    
    
    residual = filter(a, 1, x_windowed);
    VAR = var(residual);
    noise = sqrt(VAR) * randn(size(a));
    a_noisy = a + noise;
    
    
    excitation = 0.01 * randn(segment_length, 1);
    
    % Synthesize
    synth_segment = filter(1, a, excitation);
    synth_segment_noisy = filter(1, a_noisy, excitation);
 
    synthesized(start:endd) = synthesized(start:endd) + synth_segment .* win;
    synthesized_noisy(start:endd) = synthesized_noisy(start:endd) + synth_segment_noisy .* win;
end

% Post-filtering
post_filter_coeff = 0.9;
synthesized = filter(1, [1, -post_filter_coeff], synthesized);
synthesized_noisy = filter(1, [1, -post_filter_coeff], synthesized_noisy);

% results
soundsc(speech, Fs);
pause(length(speech)/Fs + 1);
soundsc(synthesized_noisy, Fs);

t = (0:length(speech)-1) / Fs;
figure;
subplot(3,1,1);
plot(t, speech);
title('Noisy Speech');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(3,1,3);
plot(t, synthesized_noisy);
title('Synthesized Speech with Noisy LPC');
xlabel('Time (s)');
ylabel('Amplitude');


% defined functions 
function a = my_lpc(x, order)
    M = order;
    N = length(x);
    r = zeros(M+1, 1);
    for k = 0:M
        r(k+1) = sum(x(1:N-k) .* x(1+k:N)) / N;
    end
    R = toeplitz(r(1:M));
    r_vec = r(2:M+1);
    w = R \ r_vec;
    a = [1; -w];
end

function noisy_signal = add_noise(signal, SNR_dB)
    signal_power = mean(signal.^2);
    noise = randn(size(signal));
    noise_power = signal_power / (10^(SNR_dB/10));
    noise = sqrt(noise_power) * noise;
    noisy_signal = signal + noise;
end


