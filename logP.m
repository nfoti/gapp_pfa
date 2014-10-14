function lp = logP(X, params)
% LOGP Compute the log joint probability of the model.
%
%   lp = logP(X, params)

log2pi = 1.83787706640935;

Xpn = params.Xpn;
Phi = params.Phi;
Psi = params.Psi;
Theta = params.Theta;
Znk = params.Znk;
Km = params.Km;
W = params.W;
Lambda = params.Lambda;
alpha_phi = params.alpha_phi;
e_psi = params.e_psi;
f_psi = params.f_psi;
Theta_a = params.Theta_a;
c0 = params.c0;
d0 = params.d0;

% Indices of topics that are being used
Kact = cellfun(@(x)(numel(x)>0), Xpn.kinds);
Kplus = sum(Kact);

N = size(Znk,1);
P = size(Phi,1);

Phi = Phi(:,Kact);
Psi = Psi(Kact,:);
Theta = Theta(Kact);
Znk = Znk(:,Kact);
W = W(:,Kact);
Lambda = Lambda(:,Kact);

lp = 0.0;

% Phi
%lp = lp + Kplus*(gammaln(P*alpha_phi) - P*gammaln(alpha_phi));
%lp = lp + (alpha_phi-1)*sum(log(Phi(:)+eps));

% Psi
%lp = lp + Kplus*N*(e_psi*log(f_psi) - gammaln(e_psi)) ...
%      + (e_psi-1)*sum(log(Psi(:)+eps)) - sum(f_psi*Psi(:));

% Theta
%lp = lp - Kplus*gammaln(Theta_a) + (Theta_a-1)*sum(log(Theta+eps)) - sum(Theta);

% psi (precision of kernels, will probably be named tau in paper)
% mus (centers of kernels)

% Lambda
%lp = lp + numel(Lambda)*(c0*log(d0) - gammaln(c0)) + ...
%     sum(sum((c0-1).*log(Lambda+eps) - d0.*Lambda));
   
% W
%[L,K] = size(W);
%L = size(W,1);
%sigma = 1./(sqrt(Lambda)+eps);
%lp = lp - sum(log(sigma(:)+eps)) - 0.5*(L+1)*Kplus*log2pi - 0.5.*sum(sum(Lambda.*(W.^2)));

% Znk
if params.dosampleZnk
    Km = Km(Kact);
    G = zeros(N,Kplus);
    for k = 1:Kplus
        G(:,k) = Km{k}*W(:,k);
        %lp = lp + sum(Znk(:,k).*normcdfln(G(:,k)+eps) + (1-Znk(:,k)).*log(1-normcdf(G(:,k))+eps));
    end
    G = normcdf(G);
else
    G = ones(N,Kplus);
end

% Likilihood - may have to write a mex function for this
%fprintf('likelihood...');
inds = X.inds;
vals = X.vals;
xx = X.vals;
NN = numel(vals);
Rate = Phi * (diag(Theta)*G'.*Psi); % Parens matter here
% for ii = 1:NN
%   r = Rate(inds(ii,1), inds(ii,2));
%   x = xx(ii);
%   lp = lp + x*log(r + eps) - r - gammaln(x+1);
% end
rr = Rate(sub2ind(size(Rate), inds(:,1), inds(:,2)));
lp = lp + sum(xx.*log(rr + eps) - rr - gammaln(xx+1));