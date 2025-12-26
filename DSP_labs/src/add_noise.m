function noisy_signal = add_noise(signal, SNR_dB)
    % Compute signal power
    signal_power = mean(signal.^2);
    
    % Generate white Gaussian noise
    noise = randn(size(signal));
    
    % Scale noise to achieve desired SNR
    noise_power = signal_power / (10^(SNR_dB/10));
    noise = sqrt(noise_power) * noise;
    
    % Add noise to signal
    noisy_signal = signal + noise;
end