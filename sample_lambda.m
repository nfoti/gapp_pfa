function lambda = sample_lambda(W, c0, d0)
%  SAMPLE_LAMBDA Sample the precisions for the weights
%  
%  Inputs:
%  W : (L+1)xK matrix of weights
%  c0 : prior shape parameter
%  d0 : prior rate parameter
%
% Returns:
% updated values of lambda

[M N] = size(W);

c = c0 + 0.5;
d = d0 + 0.5*W.^2;

lambda = gamrnd(c, 1./d, [M N]);
