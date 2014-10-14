seed = 8675309;
s = RandStream('mt19937ar', 'Seed', seed);
RandStream.setGlobalStream(s);

prefix = '~/work/data/text/sfi_abstracts/WPData20120323';

% Filter out words and documents that won't be good.
% See the function for parameters.
% Variables that are loaded by the called script
X = [];
Y = [];
T = [];
vocab = [];
preprocess_sfi_corpus;

% X contains training data, Y contains testing

P = X.P;
N = X.N;

% Number of topics
K = 200;
L = 10;

% Initialize parameters of sampler
isstatic = true;
nburn = 1000;
sampleXpn = true; % These settings define a static topic model
sampleTheta = true;
samplePhi = true;
sampleZnk = false;
sampleW = false;
sampleLambda = false;
sample_psi_inds = false;
sampleKernWidth = false;

Theta_a = 1/K;
alpha_phi = 1.01;
c0 = 1e-6;
d0 = 1e-6;

Znk = ones(N,K); % No thinning of topics
Theta = 0.01*ones(1,K); % Start all topic scores as small number
Phi = gamrnd(alpha_phi,1,P,K);
Phi = bsxfun(@rdivide, Phi, sum(Phi));
Psi = 0.01*ones(K,N);
% W has to have 1 more row than W
W = randn(L+1,K);  % Just make the following arbitrary b/c they're not used in static model
Lambda = ones(1,L);
mus = ones(L,1);
psi_inds = ones(1,K);
psi_dict = ones(1,L);
U = 1;

init = init_params_struct(X, T, 'nsamp', 1, 'thin', nburn, ...
                          'dosampleXpn', sampleXpn, ...
                          'dosampleZnk', sampleZnk, ...
                          'dosampleTheta', sampleTheta, ...
                          'dosamplePhi', samplePhi, ...
                          'dosampleW', sampleW, ...
                          'dosampleLambda', sampleLambda, ...
                          'dosample_psi_inds', sample_psi_inds, ...
                          'sampleKernWidth', sampleKernWidth, ...
                          'rng_seed', RandStream.getGlobalStream.Seed, ...
                          'Znk', Znk, ...
                          'Theta', Theta, ...
                          'Psi', Psi, ...
                          'Phi', Phi, ...
                          'W', W, ...
                          'Lambda', Lambda, ...
                          'mus', mus, ...
                          'psi_inds', psi_inds, ...
                          'psi_dict', psi_dict, ...
                          'U', U, ...
                          'Theta_a', Theta_a, ...
                          'alpha_phi', alpha_phi, ...
                          'c0', c0, 'd0', d0, ...
                          'isstatic', isstatic, ...
                          'verbose', true ...
                         );

Km = cell(1, K);
for i = 1:K
  Km{i} = randn(N,L+1);
end

lPhi = log(Phi+eps);
lTheta = log(Theta+eps);
lPsi = log(Psi+eps);

[inds vals run_sum] = sample_Xpn_kernel_meat(X.inds, X.vals, lPhi, ...
                                             lTheta, lPsi, ...
                                             Km, W, X.sum);
% run_sum contains number of rows of arrays in X, get rid of unused rows.
if run_sum < X.sum
  inds = inds(1:run_sum,:);
  vals = vals(1:run_sum);
end

return;

% Get inds corresponding to each topic
kinds = cell(K,1);
kinds(1:max(inds(:,3))) = accumarray(uint32(inds(:,3)), 1:size(inds,1), [], @(x){x});
%tmp = accumarray(uint32(inds(:,3)), 1:size(inds,1), [], @(x){x});
%nmiss = K - numel(tmp);
%for ii = 1:nmiss
%  tmp{end+1} = [];
%end

