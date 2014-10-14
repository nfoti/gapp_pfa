function Theta = sample_Theta(params)
% SAMPLE_THETA Gibbs update for topic scores, Theta.
% 
%  [Theta] = sample_Theta(params)
%
%  Input: Everything is extracted from params struct
%   Xpn : Topic contributions to counts (see sample_Xpn)
%   Znk : Thinning variables for each document
%   Theta : Topic score vector
%   aa : Shape parameter for Theta, should have 1/K factor already in it
%        for gamma process
%   
%  Returns: sampled Theta vector

Xpn = params.Xpn;
Znk = params.Znk;
Theta = params.Theta;
Psi = params.Psi;
Km = params.Km;
W = params.W;

aa = params.Theta_a;

K = size(Znk,2);

kinds = Xpn.kinds;
vals = Xpn.vals;

for k = 1:K
  
  kk = kinds{k};
  sumX = sum(vals(kk));
  %sumZPsi = Psi(k,:)*Znk(:,k);
  
  % Use p(Znk = 1) rather than Znk, i.e. use kernel GaP
  if params.dosampleZnk
    phi = normcdf(Km{k}*W(:,k));
    phi(phi == 0) = 1e-16;
    phi(phi == 1) = 1-1e-16;
  else
    phi = ones(size(Psi,2),1);
  end
  
  sumZPsi = Psi(k,:)*phi;
  
  %Theta(k) = gamrnd(sumX + aa, 1./(sumZPsi + 1));
  Theta(k) = randg(sumX+aa) ./ (sumZPsi+1);
  
end
