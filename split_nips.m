function [counts_train counts_test] = split_nips(toks, docs, test_frac, seed)
% SPLIT_NIPS Split nips corpus into training and testing set.
%
%  [counts_train counts_test] = split_nips(toks, docs, test_frac, seed)
%   toks : 1xn array with word index of k'th token if take docs as text stream
%   docs : 1xn array with doc index of k'th token in stream
%   test_frac : fraction of words in each document to use as test set (scalar)
%   seed : random seed to use
%
% Notes: Use outputs of serialize_nips.m as inputs here
%

warning('gapp_pfa:serilaize',...
  'This function has a random component, pass a seed as the second argument to fix or -1 to not permute');

setseed = 0;

if nargin < 4
  use_rand = true;
elseif nargin == 4
  if seed > -1
    setseed = 1;
    gs = RandStream.getGlobalStream;
    s = RandStream('mt19937ar', 'Seed', seed);
    RandStream.setGlobalStream(s);
    use_rand = true;
  else
    use_rand = false;
  end
end

% Hold out test_frac words from each document

D = max(docs);
nwords_pdoc = accumarray(docs',1);
ntest_wperd = ceil(test_frac.*nwords_pdoc);
ntrain_wperd = nwords_pdoc-ntest_wperd;
ntrain_el = sum(ntrain_wperd);
ntest_el = sum(ntest_wperd);
toks_train = zeros(1,ntrain_el);
docs_train = zeros(1,ntrain_el);
toks_test = zeros(1,ntest_el);
docs_test = zeros(1,ntest_el);
train_idx=1;
test_idx=1;
for d = 1:D
  
  if mod(d,100) == 0
    fprintf('doc %d\n',d);
  end
  
  inds = find(docs==d);
  dlen = numel(inds);
  if use_rand
    rp = randperm(dlen);
  else
    rp = 1:dlen;
  end
  
  ntr = ntrain_wperd(d);
  trinds = inds(rp(1:ntr));
  toks_train(train_idx:(train_idx+ntr-1)) = toks(trinds);
  docs_train(train_idx:(train_idx+ntr-1)) = docs(trinds);
  train_idx = train_idx + ntr;
  
  nte = ntest_wperd(d);
  teinds = inds(rp((ntr+1):end)); % This is supposed to be ntr
  toks_test(test_idx:(test_idx+nte-1)) = toks(teinds);
  docs_test(test_idx:(test_idx+nte-1)) = docs(teinds);
  test_idx = test_idx + nte;
  
end

% Use accumarray to construct sparse mats of training and testing data
counts_train = accumarray([toks_train' docs_train'],1);
counts_test = accumarray([toks_test' docs_test'],1);

% reset seed if we messed with it
if setseed
  RandStream.setGlobalStream(gs);
end