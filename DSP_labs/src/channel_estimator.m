% Channel estimation parameters
clc;
N = 10000;
M = 10;
iterations = 2000;
A = [2 0.1];
B = conv([1 0.9], [1 -0.8]);
num_realizations = 100;
mu = [0.01, 0.005, 0.0015];
num_mu = length(mu);

v1 = randn(N + 2000, 1);
v1 = v1(2001:end);
dn = filter(A, B, v1);

% Wiener solution
[R, p] = compute_correlations(v1, dn, M);
sigma_d = var(dn);
[w_o, J_min] = wiener_filter(R, p, sigma_d);

% True impulse response
h_true = impz(A, B, M);
h_true = h_true(1:M);

% Steepest descent
[mse_sd, w_sd] = steepest_descent(R, p, w_o, mu, iterations, M, false);
mse_sd = mse_sd + J_min;

% LMS processing
e2_lms = zeros(N, num_mu);
w_lms_avg = zeros(M, num_mu);

for mu_idx = 1:num_mu
    current_mu = mu(mu_idx);
    e2_total = zeros(N, 1);
    w_total = zeros(M, num_realizations);
    
    for r = 1:num_realizations
        v_run = randn(N + 2000, 1);
        v_run = v_run(2001:end);
        u_run = filter(A, B, v_run);
        
        % Use generalized LMS function with delay = 0 for channel estimation
        [e2_temp, w_final] = lms_adaptive_filter(v_run, u_run, zeros(M,1), current_mu, M, false, 0);
        e2_total = e2_total + e2_temp;
        w_total(:, r) = w_final;
    end
    
    e2_lms(:, mu_idx) = e2_total / num_realizations;
    w_lms_avg(:, mu_idx) = mean(w_total, 2);
end

% RLS processing
sigma_n2 = 0.01; 
[w_rls, e_rls] = rls_adaptive_filter(v1, dn, M, sigma_n2);

% Display impulse responses in console
disp('True Impulse Response:');
disp(h_true');

disp('Wiener Filter Impulse Response:');
disp(w_o');

for mu_idx = 1:num_mu
    disp(['Steepest Descent Impulse Response (mu = ', num2str(mu(mu_idx)), '):']);
    disp(w_sd(:, mu_idx)');
    
    disp(['LMS Impulse Response (mu = ', num2str(mu(mu_idx)), '):']);
    disp(w_lms_avg(:, mu_idx)');
end

disp('RLS Impulse Response:');
disp(w_rls');

% Plotting 
figure;
stem(0:M-1, h_true, 'k', 'DisplayName', 'True');
hold on;
stem(0:M-1, w_o, 'b', 'DisplayName', 'Wiener');
for mu_idx = 1:num_mu
    stem(0:M-1, w_sd(:, mu_idx), 'LineStyle', '--', 'DisplayName', sprintf('SD, mu=%.4f', mu(mu_idx)));
    stem(0:M-1, w_lms_avg(:, mu_idx), 'LineStyle', ':', 'DisplayName', sprintf('LMS, mu=%.4f', mu(mu_idx)));
end
stem(0:M-1, w_rls, 'm', 'DisplayName', 'RLS');
xlabel('Tap Index'); ylabel('Amplitude'); title('Estimated Channel Impulse Response');
legend; grid on;

% Learning Curves (including RLS)
figure;
colors = {'b', 'g', 'r'};
for mu_idx = 1:num_mu
    plot(1:iterations, mse_sd(:, mu_idx), 'Color', colors{mu_idx}, 'LineStyle', '-', 'DisplayName', sprintf('SD, mu=%.4f', mu(mu_idx)));
    hold on;
end
for mu_idx = 1:num_mu
    plot(1:N, e2_lms(:, mu_idx), 'Color', colors{mu_idx}, 'LineStyle', '--', 'DisplayName', sprintf('LMS, mu=%.4f', mu(mu_idx)));
end
xlabel('Iteration'); ylabel('MSE'); title('Learning Curves');
legend; grid on;

figure;
plot(1:N, e_rls.^2, 'm', 'DisplayName', 'RLS');
xlabel('Iteration'); ylabel('RLS'); title('Learning Curves');

