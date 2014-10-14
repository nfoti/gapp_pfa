
clear;

addpath(genpath('~/matlab/lightspeed'));

% Set seed if desired
%s = RandStream('mt19937ar','Seed',8675309);
%s = RandStream('mt19937ar','Seed',5);
s = RandStream('mt19937ar','Seed',8365); % Good demo seed
%s = RandStream('mt19937ar','Seed',rand*10000);
RandStream.setGlobalStream(s);
gs = RandStream.getGlobalStream();
fprintf('Random seed: %d\n', gs.Seed);

% Don't change this for a cluster
plot_figs = 0;

Nx = 100;
d = 8; % edge length of images
K = d;
T = 20;
N = Nx*T;
Xunq = (1:T)';
X = zeros(N,1);
offs = 2;
for i = 1:size(Xunq,1)
  X((i-1)*Nx+(1:Nx)) = i + offs;
end

% Create "Topic" (bars that are normalized)
% Create bars
bars = zeros(d*d,d);
count = 1;
for i = 1:2:d
  b = zeros(d);
  b(i,:) = ones(1,d);
  bars(:,count) = reshape(b,d*d,1);
  count = count + 1;
end
for i = 2:2:d
  b = zeros(d);
  b(:,i) = ones(d,1);
  bars(:,count) = reshape(b,d*d,1);
  count = count + 1;
end
clear count b;
Phi = bars;       % Just so names match up with papers
clear bars;
Phi = bsxfun(@rdivide, Phi, sum(Phi,1));

% % Image features
% image_part_1 = [0 1 0 0 0 0; 1 1 1 0 0 0; 0 1 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0];
% image_part_2 = [0 0 0 1 1 1; 0 0 0 1 0 1; 0 0 0 1 1 1; 0 0 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0];
% image_part_3 = [0 0 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0; 1 0 0 0 0 0; 1 1 0 0 0 0; 1 1 1 0 0 0];
% image_part_4 = [0 0 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0; 0 0 0 1 1 1; 0 0 0 0 1 0; 0 0 0 0 1 0];
% % subplot(1,4,1)
% % imagesc(image_part_1)
% % set(gca,'XTick',[])
% % set(gca,'YTick',[])
% % colormap('hot')
% % subplot(1,4,2)
% % imagesc(image_part_2)
% % set(gca,'XTick',[])
% % set(gca,'YTick',[])
% % colormap('hot')
% % subplot(1,4,3)
% % imagesc(image_part_3)
% % set(gca,'XTick',[])
% % set(gca,'YTick',[])
% % colormap('hot')
% % subplot(1,4,4)
% % imagesc(image_part_4)
% % set(gca,'XTick',[])
% % set(gca,'YTick',[])
% % colormap('hot')
% 
% Phi = [reshape(image_part_1,1,numel(image_part_1)) ; ...
%      reshape(image_part_2,1,numel(image_part_2)) ; ...
%      reshape(image_part_3,1,numel(image_part_3)) ; ...
%      reshape(image_part_4,1,numel(image_part_4))];
% Phi = Phi';
% clear image_part_1 image_part_2 image_part_3 image_part_4;

Ktrue = size(Phi,2);
d = sqrt(size(Phi,1));

psi_w = 4;
Psi = psi_w*ones(Ktrue,N);

% Generate Nx data points per covariate location

% 1d covariates are the indices stored in the rows
mus = Xunq;
psi_dict = [3];
psi_inds = ones(Ktrue,1);
L = size(mus,1);
% Use spike-and-slab to draw weights here (hier. student-t used in sampler)
%Km = computeKernMats_exp2(X, K, mus, psi_inds, psi_dict);
Km = computeKernMats_exp2(X, struct('Psi',Psi, 'mus',mus, 'psi_inds',psi_inds, 'psi_dict',psi_dict));
W = zeros(L+1,Ktrue);
H = zeros(Ktrue,N);
sigma_weights = sqrt(4);
b_a = 1;
b_b = 1;
for k = 1:Ktrue
  nu_k = betarnd(b_a,b_b);
  inds = [true ; rand(L,1) < nu_k]; % bias weight always drawn from normal
  W(inds,k) = randn(sum(inds),1).*sigma_weights;
  W(~inds,k) = 0;
  tmp = Km{k}*W(:,k);
  tmin = min(tmp);
  if tmin < 0
    tmp = tmp+abs(tmin);
  end
  tmp = tmp./max(tmp);
  g_k = tmp;
  %g_k = normcdf(tmp./max(abs(tmp)));
  %g_k = normcdf(Km{k}*W(:,k));
  g_k(g_k==0) = 1e-16;
  g_k(g_k==1) = 1-1e-16;
  H(k,:) = rand(1,N) < g_k';
end

% !! Generate data
Y = poissrnd(Phi*(H.*Psi));  % This seems better
%Y = Phi*(H.*(Psi)*psi_w);

% plot features, weights, linear functions of kernels and activation functions
if plot_figs
  figure;
  for k = 1:Ktrue
    subplot(4,Ktrue,k);
    imagesc(reshape(Phi(:,k),d,d)); colormap gray; axis image; axis off;
    title(['Feature ' num2str(k)]);
    
    subplot(4,Ktrue,Ktrue+k);
    bar(W(:,k));
    yl = ylim;
    axis([0 L+1+1 yl(1) yl(2)]);
    ylabel(['W_' num2str(k)]);
  
    %tmp = Km{k}*W(:,k);
    %g_k = tmp./max(abs(tmp));
    g_k = Km{k}*W(:,k);
    subplot(4,Ktrue,2*Ktrue+k);
    plot(X, g_k, 'LineWidth', 1.5);
    xlim([0 L+1+1]);
    ylabel(['g_' num2str(k)]);
    
    tmp = Km{k}*W(:,k);
    tmin = min(tmp);
    if tmin < 0
      tmp = tmp+abs(tmin);
    end
    tmp = tmp./max(tmp);
    % These are not the topics (names from beta process version of code)
    phi_k = tmp;
    %Phi_k = normcdf(g_k);
    phi_k(phi_k==0) = 1e-16;
    phi_k(phi_k==1) = 1-1e-16;
    subplot(4,Ktrue,3*Ktrue+k);
    plot(X, phi_k, 'r', 'LineWidth', 1.5);
    axis([0 L+1+1 0 1]);
    ylabel(['\Phi(g_' num2str(k) ')']);
    
  end
  clear yl g_k Phi_k;
  
end

% Call YY b/c X are the covariates in this script
[ii,jj,s] = find(Y);
YY = struct;
YY.inds = [ii jj];
YY.vals = s;
YY.P = size(Phi,1);
YY.N = size(Y,2);
YY.sum = sum(Y(:));
P = YY.P;
N = YY.N;

% Initialize sampler
K = 20;

isstatic = true;
nburn = 0;100;1000;
nsamp = 2;100;200;
thin = 1;

sampleXpn = true;
sampleTheta = true;
samplePhi = true;
sampleZnk = true;
sampleW = true;
sampleLambda = true;
sampleKernWidth = true;

Theta_a = 1/K;
alpha_phi = 1.01;
c0 = 1e-4;
d0 = 1e-4;
% We want the entries of psi to be big
e_psi = 10;%5;  % shape
f_psi = .5;%.5; % rate

mus = Xunq;
psi_dict = [3 .5 .05];
psi_inds = ones(K,1);
L = size(mus,1);

Theta_init = .001*ones(1,K); %0.01*ones(1,K);
% Draw columns of Phi_init from Dirichlet
Phi_init = gamrnd(alpha_phi,1,P,K);
% Set columns of Phi_init to uniform
%Phi_init = ones(P,K);
Phi_init = bsxfun(@rdivide, Phi_init, sum(Phi_init));

Psi_init = .1*ones(K,N);
W_init = zeros(L+1,K); %randn(L+1,K);
%W_init(1,:) = -1;
Lambda_init = (c0/d0)*ones(L+1,K);

verbose = true;
printfreq = 1;
reduceTopics = true;
reduceIter = 50;

smp_name = 'bars_cluster';
save_freq = 50;

init = init_params_struct(YY, X, 'nburn', nburn, 'nsamp', nsamp, ...
                          'thin', thin, ...
                          'dosampleXpn', sampleXpn, ...
                          'dosampleZnk', sampleZnk, ...
                          'dosampleTheta', sampleTheta, ...
                          'dosamplePhi', samplePhi, ...
                          'dosampleW', sampleW, ...
                          'dosampleLambda', sampleLambda, ...
                          'sampleKernWidth', sampleKernWidth, ...
                          'rng_seed', RandStream.getGlobalStream.Seed, ...
                          'Theta', Theta_init, ...
                          'Psi', Psi_init, ...
                          'Phi', Phi_init, ...
                          'W', W_init, ...
                          'Lambda', Lambda_init, ...
                          'mus', mus, ...
                          'psi_inds', psi_inds, ...
                          'psi_dict', psi_dict, ...
                          'Theta_a', Theta_a, ...
                          'alpha_phi', alpha_phi, ...
                          'e_psi', e_psi, 'f_psi', f_psi, ...
                          'c0', c0, 'd0', d0, ...
                          'isstatic', isstatic, ...
                          'verbose', verbose, 'printfreq', printfreq, ...
                          'reduceTopics', reduceTopics, ...
                          'reduceIter', reduceIter, ...
                          'smp_name', smp_name, ...
                          'save_freq', save_freq ...
                         );
   
% Set this to the number of hold-out sets
NUM_JOBS = 2;
                       
% Set up cluster params
try
  cluster = parcluster('anthill');
  anthill = 1;
catch ME
  cluster = parcluster('local');
  anthill = 0;
end
pjob = createJob(cluster);

if anthill 
  qsubargs = '-l h_rt=01:00:00 -l virtual_free=2G';  %sets max runtime to 1 hr and memory used to 2G
  set(cluster,  'IndependentSubmitFcn', {@independentSubmitFcn, qsubargs})
end

% Create jobs, in the real setting YY should be different for each one as
% each would be a different training set
for i = 1:NUM_JOBS
  createTask(pjob, @gapp_pfa_finite, 1, {YY, X, init});
end

% Run
pjob.submit();
pjob.wait();
result_params = pjob.fetchOutputs();
%params = gapp_pfa_finite(YY, X, init);

% Tear down cluster
delete(pjob);
clear pjob;

% Now do something with the result_params...
% I'm doing something really dumb and just computing the log-probability of
% the last sample of each result
lps = zeros(1,numel(result_params));
for i = 1:numel(result_params)
  % Again, a real example would use a different YY
  lps(i) = logP(YY, result_params{i});
end

return;


%%%% OLD FOR NOW

% This could be ported to the cluster...
% Plot learned stuff
figure;
plot_figs = 1;
params = result_params{1};
pidx = nsamp;
Phi_s = params.Phi_s{pidx};
Psi_s = params.Psi_s{pidx};
Theta_s = params.Theta_s{pidx};
psi_inds_s = params.psi_inds_s{pidx};
W_s = params.W_s{pidx};
Km = computeKernMats_exp2(X, struct('Psi',Psi_s, 'mus',mus, 'psi_inds',psi_inds_s, 'psi_dict',psi_dict));

% Filter out unused topics
Kact = cellfun(@(x)(numel(x)>0), params.Xpn.kinds);
Kplus = sum(Kact);

Phi_s = Phi_s(:,Kact);
Psi_s = Psi_s(Kact,:);
Theta_s = Theta_s(Kact);
Km = Km(Kact);
W_s = W_s(:,Kact);

n = size(Psi_s,1);

if plot_figs
  figure;
  for k = 1:Kplus
    subplot(4,Kplus,k);
    imagesc(reshape(Phi_s(:,k),d,d)); colormap gray; axis image; axis off;
    title(['Feature ' num2str(k)]);
    
    subplot(4,Kplus,Kplus+k);
    bar(W_s(:,k));
    yl = ylim;
    axis([0 L+1+1 yl(1) yl(2)]);
    ylabel(['W_' num2str(k)]);
  
    %tmp = Km{k}*W(:,k);
    %g_k = tmp./max(abs(tmp));
    g_k = Km{k}*W_s(:,k);
    subplot(4,Kplus,2*Kplus+k);
    plot(X, g_k, 'LineWidth', 1.5);
    xlim([0 L+1+1]);
    ylabel(['g_' num2str(k)]);
    
    phi_k = normcdf(g_k);
    phi_k(phi_k==0) = 1e-16;
    phi_k(phi_k==1) = 1-1e-16;
    subplot(4,n,3*n+k);
    plot(X, phi_k, 'r', 'LineWidth', 1.5);
    axis([0 L+1+1 0 1]);
    axis square;
    ylabel(['\Phi(g_' num2str(k) ')']);
    
  end
  clear yl g_k Phi_k;
  
end


