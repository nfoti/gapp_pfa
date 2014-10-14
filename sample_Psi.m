function [Psi] = sample_Psi(params)

  Xpn = params.Xpn;
  Theta = params.Theta;
  Znk = params.Znk;
  Km = params.Km;
  W = params.W;
  ee = params.e_psi;
  ff = params.f_psi;

  inds = Xpn.inds;
  vals = Xpn.vals;
  kinds = Xpn.kinds;
  
  [N,K] = size(Znk);
  P = Xpn.size(1);
  
  Psi = zeros(K,N);
  
  for k = 1:K
    
    kk = kinds{k};
    if numel(kk) > 0
      Xk = accumarray(inds(kk,1:2), vals(kk), [P N]);
      sumX = sum(Xk);
    else
      sumX = zeros(1,N);
    end
    
    if params.dosampleZnk
      % phi is a column vector
      phi = normcdf(Km{k}*W(:,k));
      phi(phi == 0) = 1e-16;
      phi(phi == 1) = 1-1e-16;
    else
      phi = ones(N,1);
    end
    
    %ZTheta = Znk(:,k)*Theta(k);
    % Use kernel version
    ZTheta = phi*Theta(k);
    
    %sumX = sum(Xk);
    
    %Psi(k,:) = randg(sumX + ee) ./ (ZTheta' + ff);
    Psi(k,:) = gamrnd(sumX+ee, 1./(ZTheta'+ff));
    
    %for n = 1:N
    %  Psi(k,n) = randg(sumX + ee) ./ (ZTheta(n) + ff);
    %end
    
  end
  
  
end
