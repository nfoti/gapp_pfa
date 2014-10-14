function perplex = perplexity(YY, T, params, soft)
%  PERPLEXITY Compute per-word-perplexity for gapp-pfa model
%
%   pxty = perplexity(YY, params, sof)
%
%   YY : "sparse representation of words, see gapp_pfa_finite.m for details
%   T : time stamps of documents
%   params : struct returned by gapp_pfa_finite.m
%   soft : bool indicating whether to use Znk or "soft" version of model
%   learned: temp for now, for if using learned or original data

% THIS FUNCTION PROBABLY HAS A BUG IN IT.  NOW PERPLEXITY IS COMPUTED IN
% THE SAMPLER!
error('Perplexity is computed in the sampler now.  Also, I think there is a bug in this code.  Use at your own risk!');

perplex = 0;

nsamp = params.nsamp;
inds = YY.inds;
vals = YY.vals;
N = YY.N;
ysum = YY.sum;

thetas = params.Theta_s;
phis = params.Phi_s;
psis = params.Psi_s;
Znks = params.Znk_s;
Ws = params.W_s;
psi_inds = params.psi_inds_s;
psi_dict = params.psi_dict;
mus = params.mus;

% K = size(Znks{1},2);  K no longer be fixed so loops variable length

% First compute denominator of log term b/c it's a sum over p
denom = zeros(1,N);

for s = 1:nsamp
   
   %if mod(s,2)==0
   %    fprintf('on sample %d of %d for numerator \n', s, nsamp);
   %end
    
   theta = thetas{s};
   phi = phis{s};
   psi = psis{s};
   Znk = Znks{s};
   W = Ws{s};
   psi_idx = psi_inds{s};
   K = size(phi,2);
   
   if soft == 1
       Km = computeKernMats_exp2(T, struct('Psi',psi, 'mus',mus,...
                                 'psi_inds',psi_idx, 'psi_dict',psi_dict));       
       phizes = zeros(N,K);
       for k = 1:K
           temp = normcdf(Km{k}*W(:,k));
%            temp(temp == 0) = 1e-16;
%            temp(temp == 1) = 1-1e-16;
           phizes(:,k) = temp;
       end
   end
   
   for ii = 1:numel(vals)
       p = inds(ii,1);
       n = inds(ii,2);
       inner = 0;
       
       for k = 1:K
           if soft == 1
               % CHECK THIS
               phiz = phizes(:,k); 
               inner = inner + phi(p,k)*phiz(n)*theta(k)*psi(k,n);
           elseif soft == 0
               inner = inner + phi(p,k)*Znk(n,k)*theta(k)*psi(k,n);
           end
       end
       denom(n) = denom(n) + inner;
   end
end

%Compute perplexity using above
inner_y = zeros(1,numel(vals));

for s = 1:nsamp
   
   %if mod(s,5)==0
   %    fprintf('on sample %d of %d for denom \n', s, nsamp);
   %end
    
   theta = thetas{s};
   phi = phis{s};
   psi = psis{s};
   Znk = Znks{s};
   W = Ws{s};
   psi_idx = psi_inds{s};
   K = size(phi,2);
   
   if soft == 1
       Km = computeKernMats_exp2(T, struct('Psi',psi, 'mus',mus,...
                                 'psi_inds',psi_idx, 'psi_dict',psi_dict));
       phizes = zeros(N,K);
       for k = 1:K
           temp = normcdf(Km{k}*W(:,k));
%            temp(temp == 0) = 1e-16;
%            temp(temp == 1) = 1-1e-16;
           phizes(:,k) = temp;
       end
   end
   
   for ii = 1:numel(vals)
       p = inds(ii,1);
       n = inds(ii,2);
              
       for k = 1:K
           if soft == 1
               % CHECK THIS
               phiz = phizes(:,k); 
               inner_y(ii) = inner_y(ii) + phi(p,k)*phiz(n)*theta(k)*psi(k,n);
           elseif soft == 0
               inner_y(ii) = inner_y(ii) + phi(p,k)*Znk(n,k)*theta(k)*psi(k,n);
           end
       end           
   end 
end

for ii = 1:numel(vals)
    y = vals(ii);
    n = inds(ii,2);
    perplex = perplex + y * log(inner_y(ii) / denom(n)); 
end

perplex = exp(-(1/ysum)*perplex); 
    
end
