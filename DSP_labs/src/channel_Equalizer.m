% Channel Equalizer Simulation
clc; clear; close all;

% Parameters
gamma = 5;         
M = 7;              
delta = 8;          
N = 10000;          
num_runs = 100;     
mu_values = [0.001, 0.005, 0.01];
sigma_noise2 = 5.0909e-5; 

% Channel impulse response
h = zeros(1, 2*gamma + 1);
for n = 0:2*gamma
    h(n+1) = 1 / (1 + (n - gamma)^2);
end

% Generate input signal and channel output
L = N + delta; 
a = 2 * (rand(1, L) > 0.5) - 1; 
x = filter(h, 1, a); 
noise = sqrt(sigma_noise2) * randn(1, L);
x = x + noise;

d = [zeros(1, delta) a(1:N)];

% Wiener Solution
[R, p] = compute_correlations(x, d, M);
sigma_d = var(d);
[w_wiener, J_min] = wiener_filter(R, p, sigma_d);
disp('Wiener Equalizer Coefficients:');
disp(w_wiener');

% Steepest Descent
num_iter = 1000;
[mse_sd, w_sd] = steepest_descent(R, p, w_wiener, mu_values, num_iter, M, true);

% LMS and NLMS Processing
e2_lms = zeros(L, length(mu_values));
e2_nlms = zeros(L, length(mu_values));
w_lms_avg = zeros(M, length(mu_values));
w_nlms_avg = zeros(M, length(mu_values));

for mu_idx = 1:length(mu_values)
    current_mu = mu_values(mu_idx);
    e2_lms_total = zeros(L, 1);
    e2_nlms_total = zeros(L, 1);
    w_lms_total = zeros(M, num_runs);
    w_nlms_total = zeros(M, num_runs);
    
    for r = 1:num_runs
        % Regenerate noisy signal for each run
        a_run = 2 * (rand(1, L) > 0.5) - 1;
        v_run = filter(h, 1, a_run) + sqrt(sigma_noise2) * randn(1, L);
        d_run = [zeros(1, delta) a_run(1:N)];
        
        % LMS
        w_init = zeros(M, 1);
        [e2_lms_temp, w_lms] = lms_adaptive_filter(v_run, d_run, w_init, current_mu, M, false, 0);
        e2_lms_total = e2_lms_total + e2_lms_temp;
        w_lms_total(:, r) = w_lms;
        
        % NLMS
        [e2_nlms_temp, w_nlms] = lms_adaptive_filter(v_run, d_run, w_init, current_mu, M, true, 0);
        e2_nlms_total = e2_nlms_total + e2_nlms_temp;
        w_nlms_total(:, r) = w_nlms;
    end
    
    e2_lms(:, mu_idx) = e2_lms_total / num_runs;
    e2_nlms(:, mu_idx) = e2_nlms_total / num_runs;
    w_lms_avg(:, mu_idx) = mean(w_lms_total, 2);
    w_nlms_avg(:, mu_idx) = mean(w_nlms_total, 2);
end

% Display Results
disp('Steepest Descent Equalizer Coefficients:');
for mu_idx = 1:length(mu_values)
    disp(['mu = ', num2str(mu_values(mu_idx)), ':']);
    disp(w_sd(:, mu_idx)');
end
disp('LMS Equalizer Coefficients:');
for mu_idx = 1:length(mu_values)
    disp(['mu = ', num2str(mu_values(mu_idx)), ':']);
    disp(w_lms_avg(:, mu_idx)');
end
disp('NLMS Equalizer Coefficients:');
for mu_idx = 1:length(mu_values)
    disp(['mu = ', num2str(mu_values(mu_idx)), ':']);
    disp(w_nlms_avg(:, mu_idx)');
end

% Plotting
figure;
stem(0:10, h, 'k', 'DisplayName', 'Channel Impulse Response');
hold on;
stem(0:M-1, w_wiener, 'b', 'DisplayName', 'Wiener');
for mu_idx = 1:length(mu_values)
    stem(0:M-1, w_sd(:, mu_idx), 'LineStyle', '--', 'DisplayName', sprintf('SD, mu=%.3f', mu_values(mu_idx)));
    stem(0:M-1, w_lms_avg(:, mu_idx), 'LineStyle', ':', 'DisplayName', sprintf('LMS, mu=%.3f', mu_values(mu_idx)));
    stem(0:M-1, w_nlms_avg(:, mu_idx), 'LineStyle', '-.', 'DisplayName', sprintf('NLMS, mu=%.3f', mu_values(mu_idx)));
end
xlabel('Tap Index'); ylabel('Amplitude'); title('Channel and Equalizer Responses');
legend; grid on;

figure;
for mu_idx = 1:length(mu_values)
    plot(1:num_iter, mse_sd(:, mu_idx), 'Color', 'b', 'LineStyle', '-', 'DisplayName', sprintf('SD, mu=%.3f', mu_values(mu_idx)));
    hold on;
    plot(1:L, e2_lms(:, mu_idx), 'Color', 'r', 'LineStyle', ':', 'DisplayName', sprintf('LMS, mu=%.3f', mu_values(mu_idx)));
    plot(1:L, e2_nlms(:, mu_idx), 'Color', 'g', 'LineStyle', '-.', 'DisplayName', sprintf('NLMS, mu=%.3f', mu_values(mu_idx)));
end
xlabel('Iteration'); ylabel('MSE'); title('Learning Curves');
legend; grid on;