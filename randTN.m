function Z = randTN(mu, sigma, a, b)
% RANDTN Sample from univariate truncated normal distribution.
%   Z = randTN(mu, sigma, a, b)
%
%   Notes:
%     mu, sigma, a and b should be vectors of the same length and 
%     all row or column vectors.  The returned random variables Z
%     will have the same dimensions as the input parameters.
%     
%     Also any of the parameters can be scalars if they are shared 
%     for all values.
% 
%     NO ERROR CHECKING IS DONE FOR THESE CONDITIONS AND RESULTS.
%

% This is so that the vectors can be row or column vectors.
[N1 N2] = size(mu);
phi1 = normcdf((a-mu)./sigma);
phi2 = normcdf((b-mu)./sigma);

% Should really determine if norminv is fast or if there's a better way to do
% this
u = rand(N1,N2);
Z = mu + norminv(phi1 + u.*(phi2 - phi1));
