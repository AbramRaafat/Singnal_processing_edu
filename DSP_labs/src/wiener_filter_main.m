clc; clear; close all;

% Generate signals
N = 20000;
v1 = sqrt(0.27) * randn(N, 1);
v1 = v1(5000:end);
v2 = sqrt(0.1) * randn(N, 1);
v2 = v2(5000:end);

% Define filters
a_d = [1, 0.8458];  % d(n) - 0.8458*d(n-1) = v1(n)
a_u = [1, -0.9458]; % u(n) - 0.9458*u(n-1) = v2(n)

% Generate d and u
d = filter(1, a_d, v1);
u = filter(1, a_u, d) + v2;

% Precompute variance of d
sigma_d = var(d);

% Analyze MSE for different filter orders
J_min = zeros(10, 1);
optimal_orders = zeros(10, 1);

for M = 1:10
    % Compute R and p using the correlation function
    [R, p] = compute_correlations(u, d, M);
    
    % Compute Wiener filter coefficients and MSE
    [~, J_min(M)] = wiener_filter(R, p, sigma_d);
end

% Plot results
figure;
plot(1:10, J_min, 'bo-', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
xlabel('Filter Order (M)');
ylabel('Minimum MSE (J_{min})');
title('Effect of Wiener Filter Order on Minimum MSE');
grid on;

% Find optimal filter order
[~, optimal_order] = min(J_min);
fprintf('Optimal Filter Order: M = %d\n', optimal_order);