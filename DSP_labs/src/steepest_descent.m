function [mse, w_final] = steepest_descent(R, p, w0, mu_values, num_iter, M, plot_flag)
    num_mu = length(mu_values);
    mse = zeros(num_iter, num_mu);
    w_final = zeros(M, num_mu); 
    
    if plot_flag; figure; hold on; colors = {'b','g','r','c','m'}; styles = {'-','--',':','-.','-'}; end
    for mu_idx = 1:num_mu
        mu = mu_values(mu_idx);
        w = zeros(M, 1);
        for n = 1:num_iter
            gradient = p - R * w;
            w = w + mu * gradient;
            mse(n, mu_idx) = (w - w0)' * R * (w - w0);
            if any(isnan(w)) || any(isinf(w))
                error('Numerical instability at iteration %d', n);
            end
        end
        w_final(:, mu_idx) = w; 
        if plot_flag
            plot(1:num_iter, mse(:,mu_idx), 'Color', colors{mu_idx}, 'LineStyle', styles{mu_idx}, ...
                'LineWidth', 1.5, 'DisplayName', sprintf('\\mu = %.4f', mu));
        end
    end
    if plot_flag; hold off; xlabel('Iteration'); ylabel('MSE'); title('Learning Curves'); legend; grid on; end
end