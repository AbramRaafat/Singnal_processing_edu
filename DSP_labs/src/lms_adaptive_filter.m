function [e2, w_final] = lms_adaptive_filter(u, d, w, mu, M, use_nlms, delay)

    u = u(:);
    d = d(:);
    
    N = length(u);
    e2 = zeros(N, 1);
    eps = 1e-4;
    
  
    if nargin < 7
        delay = 0;
    end
   
    for n = M + delay : N
        u_vec = u(n-delay:-1:n-delay-M+1);
        u_vec = u_vec(:);
        d_n = d(n);
        
        y = w' * u_vec;
        e = d_n - y;
        if use_nlms
            norm_u = u_vec' * u_vec + eps;
            alpha = mu / norm_u;
        else
            alpha = mu;
        end
        w = w + alpha * e * u_vec;
        e2(n) = e^2;
    end
    w_final = w;
end






