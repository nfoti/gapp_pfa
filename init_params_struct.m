function [init] = init_params_struct(X,T,varargin)
% INIT_PARAMS_STRUCT Set initial values of sampler to desired values.
%  Throws errors for missing required fields.
%
% [init] = init_params_struct(X,T,varargin)
%
% Returns params struct that can be used to initialize sampler state.

ip = inputParser;
ip.CaseSensitive = false;
ip.FunctionName = 'init_params_struct';
ip.KeepUnmatched = false;

% validation 
validate_int_pos = @(n)isnumeric(n) && n > 0 && n == round(n);
validate_int_pos0 = @(n)isnumeric(n) && n >= 0 && n == round(n);
%validate_float = @(x)isnumeric(x);
validate_float_pos = @(x)isnumeric(x) && x > 0;
validate_matrix = @(m)ismatrix(m);
validate_matvec = @(m)ismatrix(m) || isvector(m);
validate_vec = @(v)isvector(v);
% Test this one
validate_vecfloatpos = @(v)(isvector(v)||isnumeric(v)) && all(v>0);

ip.addParamValue('nburn', 0, validate_int_pos0);
ip.addParamValue('nsamp', 1, validate_int_pos);
ip.addParamValue('thin', 1, validate_int_pos);

ip.addParamValue('Znk', -1, validate_matrix);
ip.addParamValue('Theta', -1, validate_vec);
ip.addParamValue('Phi', -1, validate_matrix);
ip.addParamValue('Psi', -1, validate_matrix);
ip.addParamValue('W', -1, validate_matrix);
ip.addParamValue('Lambda', -1, validate_matrix);
ip.addParamValue('mus', -1, validate_matvec);
ip.addParamValue('psi_inds', -1, validate_vec);
ip.addParamValue('psi_dict', -1, validate_matvec);
ip.addParamValue('U', -1, validate_vecfloatpos);

ip.addParamValue('Theta_a', -1, validate_float_pos);
ip.addParamValue('alpha_phi', -1, validate_float_pos);
ip.addParamValue('e_psi', 1, validate_float_pos);
ip.addParamValue('f_psi', 1, validate_float_pos);
ip.addParamValue('c0', -1, validate_float_pos);
ip.addParamValue('d0', -1, validate_float_pos);

ip.addParamValue('alright', false, @islogical);
ip.addParamValue('isstatic', false, @islogical);
ip.addParamValue('computeperplex', false, @islogical);
ip.addParamValue('soft', false, @islogical);
ip.addParamValue('verbose', false, @islogical);
ip.addParamValue('printfreq', 0, validate_int_pos0);
ip.addParamValue('reduceTopics', false, @islogical);
ip.addParamValue('reduceIter', 100, validate_int_pos);
ip.addParamValue('smp_name', 'default', @ischar);
ip.addParamValue('save_freq', 0, validate_int_pos0);
ip.addParamValue('logger', false, @islogical);

ip.addParamValue('dosampleXpn', true, @islogical);
ip.addParamValue('dosampleZnk', true, @islogical);
ip.addParamValue('dosampleTheta', true, @islogical);
ip.addParamValue('dosamplePhi', true, @islogical);
ip.addParamValue('dosamplePsi', true, @islogical);
ip.addParamValue('dosampleW', true, @islogical);
ip.addParamValue('dosampleLambda', true, @islogical);
ip.addParamValue('sampleKernWidth', true, @islogical);

validrng_types = {'mcg16807','mlfg6331_64','mrg32k3a','mt19937ar', ...
                  'shr3cong','swb2712'};
validate_rng_type = @(x)validatestring(x,validrng_types);
ip.addParamValue('rng_type', 'mt19937ar', validate_rng_type);
ip.addParamValue('rng_seed', 0, validate_int_pos);


% Parse
ip.parse(varargin{:});
init = ip.Results;

% Detect errors
err_id = 'init_params_struct:error';
if init.Theta == -1
  error(err_id,'Theta is required');
elseif init.Phi == -1
  error(err_id,'Phi is required');
elseif init.Psi == -1
  error(err_id,'Psi is required');
elseif init.W == -1
  error(err_id,'W is required');
elseif init.Lambda == -1
  error(err_id,'Lambda is required');
elseif init.mus == -1
  error(err_id,'mus is required');
elseif init.psi_inds == -1
  error(err_id,'psi_inds is required');
elseif init.psi_dict == -1
  error(err_id,'psi_dict is required');
end

if ~isrow(init.Theta)
  error(err_id, 'Theta must be a [1xK] vector');
end
if ~ismatrix(init.Phi)
  error(err_id, 'Phi must be a [PxK] matrix');
end
if ~ismatrix(init.Psi)
  error(err_id, 'Psi must be a [KxN] matrix');
end
if ~ismatrix(init.W)
  error(err_id, 'W must be a [(L+1)xK] matrix');
end
if ~ismatrix(init.Lambda)
  error(err_id, 'Lambda must be a [(L+1)xK] matrix');
end
if ~ismatrix(init.mus)
  error(err_id, 'mus must be a [LxD] matrix');
end

if init.U == -1
  init.U = 1/size(init.Psi,1);
end

% This is indicating that we're specifying all of these variables as
% opposed to need to sample any of them to initialize
if ~init.isstatic
  nfeats = [size(init.Znk,2) numel(init.Theta) size(init.Phi,2) ...
            size(init.W,2) size(init.Lambda,2) numel(init.psi_inds)];
  if numel(unique(nfeats)) ~= 1
    error(err_id,'Must specify same number of features for model variables');
  end
  if numel(init.psi_inds) ~= nfeats(1)
    error(err_id,'psi_inds must be a K vector');
  end
end
nd = [X.N size(T,1)];
if numel(unique(nd)) ~= 1
  error(err_id,'Must specify same number of observations for data and model');
end
np = [X.P size(init.Phi,1)];
if numel(unique(np)) ~= 1
  error(err_id,'Must specify same number of words for data and model');
end


