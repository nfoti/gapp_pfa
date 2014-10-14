function [counts words years] = load_nips(fn,q)
% LOAD_NIPS Load nips data and preprocess
%
% [] = load_nips(fn,q)
%   fn : path to mat file with data
%   q  : fraction of words to keep based on tfidf distribution
%
%other funcs do all this now, just wanted the bit at the end to extract
%years data
%
%  Returns:
%    counts : PxN sparse matrix of word counts (P words, N docs)
%    words : cell array of strings, each string is a vocab word
%    years : vector with year for each paper as double
%

words=[];
counts=[];
docs_names=[];

load(fn);

clear authors_names aw_counts docs_authors;

% Corpus tfidf of each word
% tfidf = sum(counts,2) .* log(size(counts,2)./(sum(counts>0,2))+1);
% tfidf = full(tfidf);
% 
% % Keep top q% of words w.r.t tfidf
% Tw  = quantile(tfidf,1-q+eps);
% wind = tfidf > Tw;
% 
% words_short = words(wind);
% counts_short = counts(wind,:);

% clearvars -except words_short counts_short docs_names;
% words = words_short;
% counts = counts_short;
% clear words_short counts_short;

% Filter out documents that have < 200 words and any words that are never
% used
% baddocs = find(sum(counts)<200);
% counts(:,baddocs) = [];
% docs_names(baddocs) = [];
% badwords = find(sum(counts,2)<1);
% counts(badwords,:) = [];
% words(badwords) = [];
% clear baddocs badwords;

N = size(counts,2);

% extract year for each paper
years = zeros(N,1);
for i = 1:numel(docs_names)
  S = regexp(docs_names{i},'/','split');
  if numel(S) ~= 2
    keyboard;
  end
  years(i) = str2double(S{1});
end