function [toks docs] = serialize_nips(counts)
% SERIALIZE_NIPS Take count data and represent as word tokens to easily
%  construct train/test sets.
%
%  [toks docs] = serialize_nips(counts)
%    counts : PxN sparse matrix of word counts (P words, N docs)
%
%  Returns
%    toks : 1xn vector where element k is the word index of the k'th
%      overall token
%    docs : 1xn vector where element k is the doc index of the k'th overall
%      token
%
%  Notes: Use output of load_nips.m as input.
%

NT = sum(nonzeros(counts));
% toks is word index for all words in corpus (all documents strung out)
% and docs is document index for the word in the same place above
toks = zeros(1,NT);
docs = zeros(1,NT);

[ii jj ss] = find(counts);
n = numel(ii);

idx = 1;
for i = 1:n
  idx2 = idx+ss(i)-1;
  toks(idx:idx2) = ii(i);
  docs(idx:idx2) = jj(i);
  idx = idx + ss(i);
end