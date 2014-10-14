

% Minimum number of documents a word needs to appear in
docminT = 5;
% Minimum number of words a document must have
wordminT = 50;
% quantile to prune tfidf scores, keep words with scores above the quantile
% corresponding to this value
q_tfidf = 0.25;

make_test = false;
% Fraction of words in each document to hold out for testing
testfrac = 0.2;

% Load data
[X T vocab] = load_sfi_abstracts(prefix);

% Remove documents with no words
inds = sum(X) > 0;
X = X(:,inds);
T = T(inds);

% Remove stopwords
stopwords = importdata('~/work/text/common/stopwords.txt');
inds = true(numel(vocab),1);
for i = 1:numel(stopwords)
  inds(strcmp(stopwords{i},vocab)) = false;
end
X = X(inds,:);
vocab = vocab(inds);

% Prune words using tfidf, only keep words in upper (1-q_tfidf) quantile of
% distribution (i.e. q_tfidf=0.25 => keep top 75% of distribution)
tfidf = sum(X,2).*log( size(X,2)./(1 + sum(X>0,2)) );  % ./ actually necessary
tfidf_T = quantile(full(tfidf), q_tfidf);
inds = tfidf > tfidf_T;
X = X(inds,:);
vocab = vocab(inds);

% Remove documents from 1989 and 2012 since there are so few
inds = T ~= 1989 & T ~= 2012;
X = X(:,inds);
T = T(inds);

% Remove documents with few words
inds = sum(X) > wordminT;
X = X(:,inds);
T = T(inds);

% Remove words that were used in less than docT documents
docfreq = sum(X,2);
inds = docfreq > docminT;
X = X(inds,:);
vocab = vocab(inds);

[P N] = size(X);

%!!
% Create test set and store in matrix Y
if make_test
  ntwrds = zeros(1,N);
  Xnsum = zeros(1,N);
  for n = 1:N
    Xnsum(n) = sum(X(:,n));
    ntwrds(n) = floor(testfrac*Xnsum(n));
  end
  ntest_tot = sum(ntwrds);

  yinds = zeros(ntest_tot,2);
  idx = 1;

  for n = 1:N
    ii = find(X(:,n))';
    ww = zeros(Xnsum(n),1);
    rsum = 1;
    for p = ii
      ww(rsum:(rsum+X(p,n)-1)) = p;
      rsum = rsum + X(p,n);
    end
    rp = randperm(Xnsum(n));
    
    yinds(idx:(idx+ntwrds(n)-1),:) = [ww(rp(1:ntwrds(n))), n*ones(ntwrds(n),1)];
    idx = idx + ntwrds(n);
    
  end
  
  Y = sparse(yinds(:,1),yinds(:,2),1, P,N);
  
  X = X - Y;
  
  Xs = sum(X,2);
  if any(Xs) == 0
    fprintf('!!Oops, we killed the term(s):\n');
    ki = find(Xs==0)';
    for ii = ki
      fprintf('%s\n', vocab{ki});
    end
    fprintf('\n');
  end
  
end


[ii jj ss] = find(X);
X = struct;
X.P = P;
X.N = N;
X.inds = [ii jj];
X.vals = ss;
X.sum = sum(X.vals);

[ii jj ss] = find(Y);
Y = struct;
Y.P = P;
Y.N = N;
Y.inds = [ii jj];
Y.vals = ss;
Y.sum = sum(Y.vals);

clear ii jj ss P N;
clear docminT wordminT q_tfidf inds stopwords tfidf tfidf_T docfreq i D;
clear ww rsum p yinds idx Xs ki;
