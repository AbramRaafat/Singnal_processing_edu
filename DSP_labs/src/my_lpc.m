function a = my_lpc(x, order)
    % Input: x (signal segment), order (predictor order)
    % Output: a (LPC coefficients, [1, -a1, -a2, ..., -aM])
    
    M = order;
    N = length(x);
    r = zeros(M+1, 1);
    for k = 0:M
        r(k+1) = sum(x(1:N-k) .* x(1+k:N)) / N;
    end
    R = toeplitz(r(1:M));
    r_vec = r(2:M+1);
    w = R \ r_vec;
    a = [1; -w];
end