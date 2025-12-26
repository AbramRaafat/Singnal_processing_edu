function [w_rls, e_rls] = rls_adaptive_filter(u, d, M, sigma_n2)
   
    u = u(:);
    d = d(:);
    N = length(u);

    w = zeros(M, 1);         
    Q = 1e6 * eye(M);       
    e_rls = zeros(N, 1);    
    
    for n = 1:N

        if n < M
            h_n = [u(n:-1:1); zeros(M - n, 1)];
        else
            h_n = u(n:-1:n-M+1);
        end
        h_n = h_n(:); 
        

        e = d(n) - w' * h_n;
        e_rls(n) = e;
        
        denom = sigma_n2 + h_n' * Q * h_n;
        k_n = (Q * h_n) / denom;

        w = w + k_n * e;
        Q = Q - k_n * (h_n' * Q);
    end
    w_rls = w;
end