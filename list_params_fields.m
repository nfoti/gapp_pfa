function list_params_fields()
% LIST_PARAMS_FIELDS Print parameters for GaP-PFA topic model sampler.
%
% list_params_fields()
%

hline = '--------------------------------\n';

fprintf('thinned-GaP-PFA dynamic topic model\n');
fprintf('\n');

fprintf('Data\n');
fprintf(hline);
fprintf(['Assume N documents, vocab with P words and number\n' ...
         'of topics K (inferred from Theta below)\n']);
fprintf('X : PxN sparse matrix of word counts (see YY in run_simple.m)\n');
fprintf('T : Nxd matrix of covariate values for each document\n');
fprintf('params : struct to store state of sampler.\n');
fprintf('\n');
fprintf('\n');

fprintf(['To initialize all values below, specify as a field of\n' ...
         'params struct (e.g. params.Znk = Z)\n']);
fprintf('\n');

fprintf('Model parameters and latent variables\n');
fprintf(hline);
fprintf('Znk : NxK matrix of thinning indicators\n');
fprintf('Theta : 1xK vector of factor loadings\n');
fprintf('Phi : PxK matrix of "topics", Phi(:,k) is probability vector\n');
fprintf('Psi : KxN matrix of topic weights for each document\n');
fprintf('W : (L+1)xK matrix with vectors w_k of RVM in columns\n');
fprintf('Lambda : (L+1)xK matrix of precisions for RVM weights\n');
fprintf('mus : LxD matrix of covariate locaions for RMV\n');
fprintf('psi_inds : K vector with indices of atom dispersions\n');
fprintf('psi_dict : vector of unique kernel widths\n');
fprintf(['U : vector same length as psi_dict with prior prababilities\n'...
         'of the unique dispersions\n']);

fprintf('\n');
fprintf('Hyperparameters:\n');
fprintf(hline);
fprintf('Theta_a : shape parameter of gamma process\n');
fprintf('alpha_phi : parameter of symmetric Dirichlet prior on topics\n');
fprintf('e_psi : shape parameter for entries of Psi (gamma r.v.s)\n');
fprintf('f_psi : rate parameter for entries of Psi (gamma r.v.x)\n');
fprintf('c0 : shape parameter for entries of Lambda (gamma r.v.s)\n');
fprintf('d0 : rate parameter for entries of Lambda (gamma r.v.s)\n');
fprintf('\n');

fprintf('Options:\n');
fprintf(hline);
fprintf('nsamp: number of samples to gather\n');
fprintf('thin: number of samples to skip between gathering samples\n');
fprintf(['alright: (default false) skip initialization and validation ' ...
         '(e.g. starting sampler from previous run)\n']);
fprintf('dosampleXpn : flag whether to sample\n');
fprintf('dosampleZnk : flag whether to sample\n');
fprintf('dosampleTheta : flag whether to sample\n');
fprintf('dosamplePhi : flag whether to sample\n');
fprintf('dosamplePsi : flag whether to sample\n');
fprintf('dosampleW : flag whether to sample\n');
fprintf('dosampleLambda : flag whether to sample\n');
fprintf('dosample_psi_inds : flag whether to sample\n');
fprintf('sampleKernWidth : flag whether to sample\n');
fprintf('computeperplex : flag indicating whether to compute perplexity (default false)');
fprintf('soft : flag whether to use kernel (true) or Znk (false default) when computing perplexity\n');
fprintf('isstatic : flag indicating everything is initialized\n');
fprintf('rng_type : type of random stream\n');
fprintf('rng_seed : random stream seed\n');
fprintf('verbose : print progress and various state of sampler (default false)\n');
fprintf('printfreq : number of iterations to skip when printing\n');
fprintf('reduceTopics: flag whether to prune unused topics after a certain point\n');
fprintf('reduceIter: iteration after which to start pruning unused topics\n');
fprintf('smp_name: name of sampler as a string\n');
fprintf('save_freq: save results of sampler every save_freq iteations\n');
frpintf('logger: flag whether to print progress to log file (using smp_name)\n');

