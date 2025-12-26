function [R, p] = compute_correlations(u, varargin)
    % COMPUTE_CORRELATIONS Compute autocorrelation and optionally cross-correlation
    %   [R, p] = compute_correlations(u, M) computes R and p for prediction (d = u)
    %   [R, p] = compute_correlations(u, d, M) computes R for u and p for d,u
    
    if nargin < 2 || nargin > 3
        error('Invalid number of arguments. Provide either (u, M) or (u, d, M).');
    end
    
    if nargin == 2
        M = varargin{1};
        [r_uu, lags_uu] = xcorr(u, M, 'unbiased');
        r_uu_pos = r_uu(lags_uu >= 0); % r(0), r(1), ..., r(M)
        R = toeplitz(r_uu_pos(1:M)); % R = [r(0), r(1); r(1), r(0)] for M=2
        p = r_uu_pos(2:M+1);
        p = p(:); 
    else
        d = varargin{1};
        M = varargin{2};
        [r_uu, lags_uu] = xcorr(u, M-1, 'unbiased');
        r_uu_pos = r_uu(lags_uu >= 0);
        R = toeplitz(r_uu_pos(1:M));
        [r_du, lags_du] = xcorr(d, u, M-1, 'unbiased');
        p = r_du(lags_du >= 0)'; % p = [r_du(0), r_du(1), ...]'
        p = p(:); 
    end
end