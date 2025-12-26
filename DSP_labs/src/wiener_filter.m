function [w_o, J_min] = wiener_filter(R, p, sigma_d)
    % Solve Wiener-Hopf equation: R * w_o = p
    w_o = R \ p;
    
    %minimum MSE
    J_min = sigma_d - p' * w_o;
end


