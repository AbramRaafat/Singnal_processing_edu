clc; clear; close all;

% Parameters
a1 = 0.1;
a2 = -0.8;
sigma_v_initial = 1;
N = 10000;
M = 2;
mu = 0.005; 
num_realizations = 1000;

v = sigma_v_initial * randn(N, 1);
u = filter(1, [1, a1, a2], v);
var_u = var(u);
sigma_v = sigma_v_initial * sqrt(1 / var_u);

disp('Calculated sigma_v:'); disp(sigma_v^2);
disp('Variance of u(n) after scaling:'); disp(var(filter(1, [1, a1, a2], sigma_v * randn(N, 1))));

% Wiener filter
v = sigma_v * randn(N, 1);
u = filter(1, [1, a1, a2], v);
[R, p] = compute_correlations(u, M);

disp('R:'); disp(R);
disp('p:'); disp(p);

sigma_d = var(u);
[w_o, J_min] = wiener_filter(R, p, sigma_d);
disp('Optimal weights (w_o):'); disp(w_o');
disp('J_min:'); disp(J_min);

% LMS and NLMS processing using the new function
[e2_lms, e2_nlms, w_lms_avg, w_nlms_avg] = lms_solution(1, [1, a1, a2], sigma_v, N, M, mu, num_realizations, 1);

disp('LMS average weights:'); disp(w_lms_avg');
disp('NLMS average weights:'); disp(w_nlms_avg');

% Steepest descent
mu_values = [0.05];
num_iter = 10000;
mse_sd = steepest_descent(R, p, w_o, mu_values, num_iter, M, 0);
mse_sd = mse_sd + J_min;

% Plotting
figure;
plot(1:N, e2_lms, 'b', 'DisplayName', 'LMS');
xlabel('Iteration'); ylabel('MSE'); legend; title('LMS Learning Curve'); grid on; ylim([0, 0.5]);

figure;
plot(1:N, e2_nlms, 'r', 'DisplayName', 'NLMS');
xlabel('Iteration'); ylabel('MSE'); legend; title('NLMS Learning Curve'); grid on; ylim([0, 0.5]);

figure;
plot(1:num_iter, mse_sd, 'k', 'DisplayName', 'Steepest Descent');
xlabel('Iteration'); ylabel('MSE'); legend; title('Steepest Descent Learning Curve'); grid on; ylim([0, 0.5]);

figure;
plot(1:N, e2_lms, 'b', 'DisplayName', 'LMS');
hold on;
plot(1:N, e2_nlms, 'r', 'DisplayName', 'NLMS');
plot(1:min(N, num_iter), mse_sd(1:min(N, num_iter)), 'k', 'DisplayName', 'Steepest Descent');
xlabel('Iteration'); ylabel('MSE'); legend; title('Learning Curves Comparison'); grid on; ylim([0, 0.5]);





