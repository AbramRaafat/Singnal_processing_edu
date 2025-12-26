function [e2_lms_avg, e2_nlms_avg, w_lms_avg, w_nlms_avg] = lms_solution(A, B, sigma_v, N, M, mu, num_realizations, delay)
    e2_lms = zeros(N, 1);
    e2_nlms = zeros(N, 1);
    w_lms_final = zeros(M, num_realizations);
    w_nlms_final = zeros(M, num_realizations);
    
    for r = 1:num_realizations
        v = sigma_v * randn(N + 2000, 1);
        v = v(2001:end);
        u = filter(A, B, v);
        
        % LMS (delay = 1 for AR prediction)
        w_lms = zeros(M, 1);
        [e2_lms_temp, w_lms] = lms_adaptive_filter(u, u, w_lms, mu, M, false, delay);
        e2_lms(M+1:N) = e2_lms(M+1:N) + e2_lms_temp(M+1:N);
        w_lms_final(:, r) = w_lms;
        
        % NLMS (delay = 1 for AR prediction)
        w_nlms = zeros(M, 1);
        [e2_nlms_temp, w_nlms] = lms_adaptive_filter(u, u, w_nlms, mu, M, true, delay);
        e2_nlms(M+1:N) = e2_nlms(M+1:N) + e2_nlms_temp(M+1:N);
        w_nlms_final(:, r) = w_nlms;
    end

    e2_lms_avg = e2_lms / num_realizations;
    e2_nlms_avg = e2_nlms / num_realizations;
    w_lms_avg = mean(w_lms_final, 2);
    w_nlms_avg = mean(w_nlms_final, 2);
end