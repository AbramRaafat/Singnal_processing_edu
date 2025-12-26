clc;
clear;
close all;

% Generate signals
N = 10000;
v1 = sqrt(0.27) * randn(N, 1);
v2 = sqrt(0.1) * randn(N, 1);

a_d = [1, 0.8458];
d = filter(1, a_d, v1);
a_u = [1, -0.9458];
u = filter(1, a_u, d) + v2;
sigma_d = var(d);
M = 4;


[R, p] = compute_correlations(u, d, M);
[w0, ~] = wiener_filter(R, p, sigma_d);

% Compute step-size limits

lambda = eig(R);
lambda_max = max(lambda);
mu_max = 2 / lambda_max;
fprintf('Maximum eigenvalue (λ_max): %.4f\n', lambda_max);
fprintf('2/λ_max: %.4f\n', mu_max); 

% Step sizes to test
mu_values = [0.1, 0.2, 0.3, 0.4, 0.5];
num_iter = 100;

% learning curves with ploting 
plot_flag = true;
mse = steepest_descent(R, p, w0, mu_values, num_iter, M, plot_flag);



%results
fprintf('Optimal Wiener filter coefficients:\n');
disp(w0');

for mu_idx = 1:length(mu_values)
    mu = mu_values(mu_idx);
    w_final = zeros(M, 1);
    for n = 1:num_iter
        w_final = w_final + mu * (p - R * w_final);
    end
    fprintf('Final weights for mu = %.1f:\n', mu);
    disp(w_final');
end

disp('R:')
disp(R)

