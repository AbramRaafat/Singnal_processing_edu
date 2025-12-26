function [estimates, covariances] = kalman_filter(F, C, Q, R, measurements, x0, P0)
    n_states = length(x0);
    n_steps = size(measurements, 2);
    
    x = x0;
    P = P0;
    estimates = zeros(n_states, n_steps);
    covariances = zeros(n_states, n_states, n_steps);
    
    for t = 1:n_steps
        % Prediction step
        x_pred = F * x;
        P_pred = F * P * F' + Q;
        
        % Kalman gain
        S = C * P_pred * C' + R;
        K = P_pred * C' / S;
        
        % Update step
        y = measurements(:,t);
        e = y - C * x_pred;
        x = x_pred + K * e;
        P = (eye(n_states) - K * C) * P_pred;
        
        estimates(:,t) = x;
        covariances(:,:,t) = P;
    end
end